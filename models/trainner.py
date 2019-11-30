import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from tqdm import tqdm

from .ed import EDSeg


class Trainner(object):
    def __init__(self, opt, saver, summary):
        self.opt = opt
        self.saver = saver
        self.summary = summary
        self.setup_models()
        print('initialize trainner')

    def setup_models(self):
        self.criteriasMSE = nn.MSELoss()
        self.model = EDSeg(self.opt).cuda(self.opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.schedule = MultiStepLR(self.optimizer, milestones=[1600], gamma=self.opt.gamma)

        if self.opt.device_count > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.devices)

    def train_iter(self, images, labels):
        self.model.train()
        output = self.model(images)
        loss = self.criteriasMSE(output, images)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.schedule.step()

        if self.global_steps % self.opt.v_freq == 0:
            self.summary.add_scalar('loss', loss.item(), self.global_steps)
            self.summary.train_image(images, output, labels, self.global_steps)

    def train_epoch(self, dataloader):
        iterator = tqdm(dataloader,
                        leave=True,
                        dynamic_ncols=True)
        self.dataset_len = len(dataloader)
        for i, data in enumerate(iterator):
            iterator.set_description(f'Epoch[{self.epoch}/{self.opt.epochs}]')
            self.global_steps = self.epoch * self.dataset_len + i

            if isinstance(data, tuple):
                images = data[0]
                labels = data[1]
            else:
                labels = None
                images = data

            images = images.to(self.opt.device)
            labels = labels.to(self.opt.device) if labels else None

            self.train_iter(images, labels)

    def train(self, train_dataloader, valid_dataloader):
        self.epoch = 0
        while self.epoch < self.opt.epochs:
            self.train_epoch(train_dataloader)
            self.validate(valid_dataloader)
            self.epoch += 1

    def validate(self, dataloader):
        self.model.eval()
        errs = []
        output = None

        for data in dataloader:
            if isinstance(data, tuple):
                images = data[0]
                labels = data[1]
            else:
                labels = None
                images = data

            images = images.to(self.opt.device)
            labels = labels.to(self.opt.device) if labels else None
            
            output = self.model(images)
            err = self.criteriasMSE(output, images)
            errs.append(err)

        mean_err = torch.tensor(errs).mean()

        self.summary.val_image(images, output, labels, self.global_steps)
        self.save_checkpoint(mean_err)

    def save_checkpoint(self, err):
        state = {'epoch': self.epoch,
                'err': err,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'schedule': self.schedule.state_dict()}
        self.saver.save_checkpoint(state)

    def load_checkpoint(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.schedule.load_state_dict(state['schedule'])
        self.epoch = state['epoch']

    def resume(self):
        if self.opt.resume:
            if self.opt.resume_best:
                state = self.saver.load_best()
            elif self.opt.resume_latest:
                state = self.saver.load_latest()
            elif self.opt.resume_epoch is not None:
                state = self.saver.load_epoch(self.opt.resume_epoch)
            else:
                raise RuntimeError('resume settings error, please check your config file.')
            self.load_checkpoint(state)
        else:
            print('resume not enabled, pass')

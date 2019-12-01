import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch import autograd
from tqdm import tqdm

from .ed import EDSeg
from .networks import Discriminator


class Trainner(object):
    def __init__(self, opt, saver, summary):
        self.opt = opt
        self.saver = saver
        self.summary = summary
        self.global_steps = 0
        self.setup_models()
        print('initialize trainner')

    def setup_models(self):
        self.criteriasMSE = nn.MSELoss()
        self.generator = EDSeg(self.opt).cuda(self.opt.device)
        self.discriminator = Discriminator(self.opt).cuda(self.opt.device)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lrG)
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lrD)
        self.scheduleG = MultiStepLR(self.optimizerG, milestones=[1600], gamma=self.opt.gammaG)
        self.scheduleD = MultiStepLR(self.optimizerD, milestones=[1600], gamma=self.opt.gammaD)

        if self.opt.device_count > 1:
            self.generator = nn.DataParallel(self.generator, device_ids=self.opt.devices)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=self.opt.devices)

    def train_iter(self, images, labels):
        self.generator.train()
        self.discriminator.train()

        real_valid = autograd.Variable(torch.ones(images.size(0), 1)).to(self.opt.device)
        fake_valid = autograd.Variable(torch.zeros(images.size(0), 1)).to(self.opt.device)
        
        ##########################
        #  Update Discriminator
        ##########################
        self.discriminator.zero_grad()
        fake = self.generator(images)
        real_result = self.discriminator(images)
        fake_result = self.discriminator(fake.detach())

        real_loss = self.criteriasMSE(real_result, real_valid)
        fake_loss = self.criteriasMSE(fake_result, fake_valid)

        loss_D = (real_loss + fake_loss) * 0.5

        loss_D.backward()

        self.optimizerD.step()

        ##########################
        #  Update Generator
        ##########################
        self.generator.zero_grad()
        fake_result = self.discriminator(fake)
        loss_G = self.criteriasMSE(fake_result, real_valid)

        loss_G.backward()

        self.optimizerG.step()

        self.scheduleD.step()
        self.scheduleG.step()

        #########################
        #  Visualizetion
        #########################
        if self.global_steps % self.opt.v_freq == 0:
            lr_G = self.optimizerG.param_groups[0]['lr']
            lr_D = self.optimizerD.param_groups[0]['lr']

            self.summary.add_scalar('loss/generator', loss_G.item(), self.global_steps)
            self.summary.add_scalar('loss/discriminator', loss_D.item(), self.global_steps)
            self.summary.add_scalar('lr/generator', lr_G, self.global_steps)
            self.summary.add_scalar('lr/discriminator', lr_D, self.global_steps)
            self.summary.train_image(images, fake, labels, self.global_steps)

    def train_epoch(self, dataloader):
        iterator = tqdm(dataloader,
                        leave=True,
                        dynamic_ncols=True)
        self.dataset_len = len(dataloader)
        for i, data in enumerate(iterator):
            iterator.set_description(f'Epoch[{self.epoch}/{self.opt.epochs}|{self.global_steps}]')
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
        self.generator.eval()
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
            
            output = self.generator(images)
            err = self.criteriasMSE(output, images)
            errs.append(err)

        mean_err = torch.tensor(errs).mean()

        self.summary.add_scalar('validate_error', mean_err, self.global_steps)
        self.summary.val_image(images, output, labels, self.global_steps)
        self.save_checkpoint(mean_err)

    def save_checkpoint(self, err):
        state = {'epoch': self.epoch,
                'err': err,
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimizerG': self.optimizerG.state_dict(),
                'optimizerD': self.optimizerD.state_dict(),
                'scheduleG': self.scheduleG.state_dict(),
                'scheduleD': self.scheduleD.state_dict()}
        self.saver.save_checkpoint(state)

    def load_checkpoint(self, state):
        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.optimizerG.load_state_dict(state['optimizerG'])
        self.optimizerD.load_state_dict(state['optimizerD'])
        self.scheduleG.load_state_dict(state['scheduleG'])
        self.scheduleD.load_state_dict(state['scheduleD'])
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

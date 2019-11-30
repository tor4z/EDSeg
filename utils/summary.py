import os
import numpy as np
from torchvision.utils import make_grid
from torch.utils import tensorboard


class Summary(object):
    def __init__(self, opt):
        self.dir = os.path.join(opt.summary_dir, opt.dataset, opt.runtime_id)
        self.writer = tensorboard.SummaryWriter(log_dir=self.dir)
        self.disp_images = opt.disp_images
        print('initialize summary')

    def add_image(self, tag, image, global_steps):
        self.writer.add_image(tag, image, global_steps)

    def train_image(self, input, output, label, global_steps):
        # display input image
        grid_image = make_grid(input[:self.disp_images, :, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/input', grid_image, global_steps)

        # display output image
        grid_image = make_grid(output[:self.disp_images, :, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/output', grid_image, global_steps)

        if label:
            # display label image
            grid_image = make_grid(label[:self.disp_images, :, :, :].data,
                                    self.disp_images, normalize=True)
            self.add_image('train/label', grid_image, global_steps)

    def val_image(self, input, output, label, global_steps):
        # display input image
        grid_image = make_grid(input[:self.disp_images, :, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('val/input', grid_image, global_steps)

        # display output image
        grid_image = make_grid(output[:self.disp_images, :, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('val/output', grid_image, global_steps)

        if label:
            # display label image
            grid_image = make_grid(label[:self.disp_images, :, :, :].data,
                                    self.disp_images, normalize=True)
            self.add_image('val/label', grid_image, global_steps)

    def add_text(self, tag, text, global_steps):
        self.writer.add_text(tag, text, global_steps)
        self.flush()

    def add_scalars(self, tag, scalars, global_steps):
        self.writer.add_scalars(tag, scalars, global_steps)
        self.flush()

    def add_scalar(self, tag, value, global_steps):
        self.writer.add_scalar(tag, value, global_steps)
        self.flush()

    def flush(self):
        self.writer.flush()

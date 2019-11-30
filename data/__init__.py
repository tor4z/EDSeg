import os
import copy
import glob

from numpy import random
from .hela import Hela


datasets = {
    'hela': Hela}


class SingleValidation(object):
    def __init__(self, opt):
        self.valid = opt.sin_valid
        self.opt = opt

        wild_filename = f'*.{opt.suffix}'
        self.images = sorted(glob.glob(os.path.join(opt.train_path, wild_filename)))
        if opt.label_path is not None:
            self.labels = sorted(glob.glob(os.path.join(opt.label_path, wild_filename)))
        else:
            self.labels = []

        self.dataset_cls = datasets[opt.dataset]
        self.len = len(self.images)

        self.curr_valid = 0

    def random_valid(self):
        while self.curr_valid < self.valid:
            index = random.randint(0, self.len-1, size=1)[0]
            images = copy.copy(self.images)
            valid_images = images.pop(index)
            train_images = images

            if self.labels:
                labels = copy.copy(self.labels)
                valid_labels = labels.pop(index)
                train_labels = labels
            else:
                valid_labels = []
                train_labels = []
            
            train_dataset = self.dataset_cls(self.opt, train_images, train_labels)
            valid_dataset = self.dataset_cls(self.opt, valid_images, valid_labels)

            yield train_dataset, valid_dataset
            self.curr_valid += 1
        return

    def full_valid(self):
        while self.curr_valid < self.len:
            images = copy.copy(self.images)
            valid_images = images.pop(self.curr_valid)
            train_images = images

            if self.labels:
                labels = copy.copy(self.labels)
                valid_labels = labels.pop(self.curr_valid)
                train_labels = labels
            else:
                valid_labels = []
                train_labels = []
            
            train_dataset = self.dataset_cls(self.opt, train_images, train_labels)
            valid_dataset = self.dataset_cls(self.opt, valid_images, valid_labels)
            
            yield train_dataset, valid_dataset
            self.curr_valid += 1
        return
    
    def no_valid(self):
        yield self.dataset_cls(self.opt, self.images, self.labels), None

    def gen_dataset(self):
        if self.valid == 0:
            return self.no_valid()
        elif self.valid is None:
            return self.full_valid()
        else:
            return self.random_valid()


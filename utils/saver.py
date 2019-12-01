import os
import shutil
import torch
import glob

from .config import Config

class Saver(object):
    CONFIG_FILE = 'cfg.pth'
    BEST_FILE = 'best.pth'
    LATEST_FILE = 'latest.pth'
    CHECKPOINT_FILE = 'checkpoint_{}.pth'

    def __init__(self, opt):
        self.force = opt.force
        self.enable = opt.saver_enable

        if self.enable:
            self.dir = os.path.join(opt.saver_dir, opt.dataset, opt.runtime_id)
            if os.path.exists(self.dir):
                if self.force:
                    shutil.rmtree(self.dir)
                else:
                    raise RuntimeError('{} already exists.'.format(self.dir))
            else:
                try:
                    os.makedirs(self.dir)
                except Exception as e:
                    raise RuntimeError(str(e))
            
            self.best_filename = os.path.join(self.dir, self.BEST_FILE)
            self.latest_filename = os.path.join(self.dir, self.LATEST_FILE)
            self.config_filename = os.path.join(self.dir, self.CONFIG_FILE)

            self.best_err = 1000000

            self.save_config(opt)
            print('initialize saver')

    def gen_checkpoint_filename(self, epoch):
        if epoch < 0:
            raise ValueError('ecoch={}, epoch should be >= 0'.format(epoch))
        path = os.path.join(self.dir, self.CHECKPOINT_FILE.format(epoch))
        if os.path.exists(path):
            if self.force:
                os.remove(path)
            else:
                raise RuntimeError('{} already exists.'.format(path))
        return path

    def save_checkpoint(self, state):
        if not self.enable:
            return
        epoch = state['epoch']
        err = state['err']
        path = self.gen_checkpoint_filename(epoch)
        self.save(state, path)
        self.save_latest(path)
        if err < self.best_err:
            self.save_best(path)

    def load_epoch(self, epoch):
        path = self.gen_checkpoint_filename(epoch)
        state = self.load(path)
        return state

    def save_latest(self, path):
        if os.path.exists(self.latest_filename):
            os.remove(self.latest_filename)
        shutil.copyfile(path, self.latest_filename)

    def load_latest(self):
        state = self.load(self.latest_filename)
        return state

    def save_best(self, path):
        if os.path.exists(self.best_filename):
            os.remove(self.best_filename)
        shutil.copyfile(path, self.best_filename)

    def load_best(self):
        state = self.load(self.best_filename)
        return state

    def save_config(self, opt):
        if not self.enable:
            return
        cfg = opt.dump()
        self.save(cfg, self.config_filename)

    def load_config(self, path=None):
        if path is None:
            path = self.config_filename
        cfg = self.load(path)
        opt = Config()
        opt.load(cfg)
        return opt

    def check_dir(self, path):
        if not os.path.exists(path):
            raise RuntimeError('{} not exists.'.format(path))

    def save(self, obj, path):
        torch.save(obj, path)

    def load(self, obj, path):
        if not self.enable:
            raise RuntimeError('Saver not enabled.')
        self.check_dir(path)
        state = torch.load(obj, path)
        self.post_load(state)
        return state

    def post_load(self, state):
        self.best_err = state['err']

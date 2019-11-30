import yaml
import datetime
import torch


class Config(object):
    def __init__(self, cfg=None):
        if cfg is not None:
            with open(cfg) as file:
                self.cfg = yaml.load(file, Loader=yaml.FullLoader)
            self.post_config()
            self.init_attr()

    def init_attr(self):
        for k, v in self.cfg.items():
            self.__setattr__(k, v)

    def post_config(self):
        if self.cfg['devices']:
            self.cfg['cuda'] = True
            self.cfg['device'] = torch.device('cuda:{}'.format(self.cfg['devices'][0]))
        else:
            self.cfg['cuda'] = False
            self.cfg['device'] = torch.device('cpu')

        if self.cfg['resume'] and not self.cfg['runtime_id']:
            raise RuntimeError('when resume is enabled, runtime_id should be specified.')
        if self.cfg['disp_images'] > self.cfg['batch_size']:
            raise ValueError('disp_images should be less than batch_size.')

        if not self.cfg['runtime_id']:
            dtate_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.cfg['runtime_id'] = dtate_str

        self.cfg['device_count'] = len(self.cfg['devices']) or 1

    def dump(self):
        return self.cfg

    def load(self, cfg):
        self.cfg = cfg
        self.init_attr()


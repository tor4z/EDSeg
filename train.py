import argparse
from torch.utils.data import DataLoader

from data import SingleValidation
from models.trainner import Trainner
from utils.saver import Saver
from utils.summary import Summary
from utils.config import Config


def main(opt):
    print('starting.')
    saver = Saver(opt)
    summary = Summary(opt)
    sv = SingleValidation(opt)

    for train, valid in sv.gen_dataset():
        train_dataloader = DataLoader(train,
                                batch_size=opt.batch_size,
                                num_workers=opt.num_workers,
                                pin_memory=opt.pin_memory,
                                shuffle=opt.shuffle)
        if valid:
            valid_dataloader = DataLoader(valid, num_workers=opt.num_workers,
                                          pin_memory=opt.pin_memory)
        else:
            valid_dataloader = None

        trainner = Trainner(opt, saver, summary)
        trainner.train(train_dataloader, valid_dataloader)

    print('Finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Encoder Decoder')
    parser.add_argument('--cfg', type=str, help='config file path.', default='cfg.yaml')
    args = parser.parse_args()

    opt = Config(args.cfg)

    main(opt)


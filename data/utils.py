from libtiff import TIFF
import numpy as np
import torch


def read_tiff(path):
    tif = TIFF.open(path)
    images = []
    for frame in tif.iter_images():
        images.append(frame)
    return torch.Tensor(images)


def imread(path):
    suffix = path.split('.')[-1]
    if suffix == 'tif':
        return read_tiff(path)
    else:
        raise NotImplementedError(f'Cann\'t read {path}.')

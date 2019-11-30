from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import imread


class Hela(Dataset):
    def __init__(self, opt, images, labels):
        if not isinstance(images, list):
            images = [images]
        if not isinstance(labels, list):
            labels = [labels]

        self.images = []
        self.labels = []

        self.len = len(images)
        self.labels_len = len(labels)

        # Loading images
        print(f'Loading {self.len} images...')
        iterator = tqdm(images,
                        leave=True,
                        dynamic_ncols=True)
        for i, image_path in enumerate(iterator):
            iterator.set_description(f'Image[{i}/{self.len}]')
            image = imread(image_path)
            self.images.append(image)
        
        # Loading labels
        print(f'Loading {self.labels_len} labels...')
        iterator = tqdm(labels,
                        leave=True,
                        dynamic_ncols=True)
        for i, label_path in enumerate(iterator):
            iterator.set_description(f'Image[{i}/{self.labels_len}]')
            label = imread(label_path)
            self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]

        if self.labels:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return self.len


import os

import numpy as np
import PIL.Image as Image
import torch
import torchvision as tv


class Flipper:
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, x):
        if np.random.binomial(1, self._p):
            x = (x[0].transpose(Image.FLIP_LEFT_RIGHT), x[1][::-1].copy())
        return x

    
class Jitter:
    def __init__(self, *args, **kwargs):
        self._jitter = tv.transforms.ColorJitter(*args, **kwargs)

    def __call__(self, x):
        return self._jitter(x[0]), x[1]


def to_numpy(x):
    global A
    A = np.array(x[0]).transpose((2, 0, 1)), x[1]
    return np.array(x[0]).transpose((2, 0, 1)), x[1]


def _generate_mask(shape):
    return np.ones(shape, dtype=np.float32)

    
class _CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, transform=None):
        self._input = [
            os.path.join(input_dir, x)
            for x in os.listdir(input_dir)
        ]
        self._transform = transform or to_numpy

    def __len__(self):
        return len(self._input)

    def __getitem__(self, index):
        img = Image.open(self._input[index])
        return self._transform((img, _generate_mask(img.size[::-1])))


def make_default_transform():
    return tv.transforms.Compose([
        Flipper(),
        Jitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        to_numpy
    ])


def make_dataloader(input_dir, transform='default', *args, **kwargs):
    if transform == 'default':
        transform = make_default_transform()
    dataset = _CelebaDataset(input_dir, transform)
    return torch.utils.data.DataLoader(dataset, *args, **kwargs)

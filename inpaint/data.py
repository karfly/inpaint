import math
import os

import numpy as np
import PIL.Image as Image
import torch
import torchvision as tv
import jsonlines

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class _Flipper:
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, x):
        if np.random.binomial(1, self._p):
            x = (x[0].transpose(Image.FLIP_LEFT_RIGHT), x[1][::-1].copy())
        return x


class _Jitter:
    def __init__(self, *args, **kwargs):
        self._jitter = tv.transforms.ColorJitter(*args, **kwargs)

    def __call__(self, x):
        return self._jitter(x[0]), x[1]


def _final_transform(x):
    # shape must be sqaure with a side that is power of 2
    assert x[0].size[0] == x[0].size[1]
    assert not math.modf(math.log2(x[0].size[0]))[0]
    return np.array(x[0], dtype=np.float32).transpose((2, 0, 1)) / 255.0, x[1]


def _generate_dummy_mask(shape):
    return np.ones(shape, dtype=np.float32)


def _generate_random_mask(shape):
    return (np.random.rand(*shape) > 0.2).astype('float32')


class _MaskGenerator:
    def __init__(self, masks_dir, n_images_by_file=10000):
        self.strokes = []
        n_images_by_file = 10000

        for name in os.listdir(masks_dir):
            reader = jsonlines.open(os.path.join(masks_dir, name))

            i = 0
            for item in reader:
                self.strokes.extend(item['drawing'])

                i += 1
                if i > n_images_by_file:
                    break

    def __call__(self, shape, min_stroke_width=8):
        target_width, target_height = shape
        n_strokes = int(np.random.normal(5, 1))
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')

        def central_crop(image, new_width, new_height):
            image = image.copy()

            width, height = image.size

            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            image = image.crop((left, top, right, bottom))

            return image

        for _ in range(n_strokes):
            stroke_index = np.random.randint(0, len(self.strokes))
            stroke_width = np.random.randint(
                min_stroke_width, min_stroke_width + 2
            )
            stroke = self.strokes[stroke_index][
                :int(len(self.strokes[stroke_index]) * 1)
            ]
            ax.plot(*stroke, color='black', lw=stroke_width)

        canvas.draw()       # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(
            canvas.tostring_rgb(),
            dtype='uint8'
        ).reshape(int(height), int(width), 3)
        image = Image.fromarray(image)

        image = image.resize(
            (int(1.25 * target_width), int(1.25 * target_height))
        )
        image = central_crop(image, target_width, target_height)
        image = image.convert('1')

        return np.array(image).astype('float32')


class _CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, masks_dir, transform=_final_transform):
        self._input = [
            os.path.join(input_dir, x)
            for x in os.listdir(input_dir)
        ]
        self._transform = transform
        if masks_dir is not None:
            self._mask_generator = _MaskGenerator(masks_dir)
        else:
            self._mask_generator = _generate_random_mask

    def __len__(self):
        return len(self._input)

    def __getitem__(self, index):
        img = Image.open(self._input[index])
        return self._transform((img, self._mask_generator(img.size[::-1])))


def _make_default_transform():
    return tv.transforms.Compose([
        _Flipper(),
        _Jitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        _final_transform
    ])


def make_dataloader(input_dir, masks_dir, transform='default', *args, **kwargs):
    """Make data loader: input images and generated masks.

    Parameters
    ----------
    input_dir : str
        path to the folder containing input images (CelebaDataset HQ)

    masks_dir : str
        path to the folder containing simple images (to generate masks from) in .ndjson format (Quick Draw Dataset)
        (https://github.com/googlecreativelab/quickdraw-dataset)

    transform : str
        type of data augmentation. currently supported types: "default"

    Returns
    -------
    data_loader : DataLoader
        data loader yielding input (transformed) image and generated mask
    """
    if transform == 'default':
        transform = _make_default_transform()
    dataset = _CelebaDataset(input_dir, masks_dir, transform)
    return torch.utils.data.DataLoader(dataset, *args, **kwargs)

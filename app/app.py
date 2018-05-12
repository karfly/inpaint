#!/usr/bin/env python3

import argparse
import base64
import io
import os
import sys
import random

import numpy as np
from PIL import Image
from flask import Flask, flash, render_template, redirect, request, send_file

sys.path.append('..')
from inpaint import make_inpainter


STATIC_DIR = 'static'
RANDOM_IMAGES_DIR = None
DEFAULT_RANDOM_IMAGES_DIR = os.path.join(STATIC_DIR, 'random_images')
INPAINTER = None


app = Flask(__name__)


def setup_app(
    model_state_dict=None,
    random_images_dir=DEFAULT_RANDOM_IMAGES_DIR,
    device='cpu'
):
    """Configure Flask application.
    
    Parameters
    ----------
    model_state_dict : str
        Path to a pytorch state dict compatible with inpaint.InpaintNet

    random_images_dir : str
        Path to a dir containing sample images

    device : str
        Device compatible with torch.device. Determines the device where
        an instance of inpaint.InpaintNet will run.
        Examples: 'cpu', 'gpu', 'gpu:1'

    Returns
    -------
    app : flask.Flask
    """
    global INPAINTER
    global RANDOM_IMAGES_DIR
    INPAINTER = make_inpainter(model_state_dict)
    INPAINTER.set_device(device)
    RANDOM_IMAGES_DIR = random_images_dir
    return app


def extract_image(request, field_name, new_shape=None):
    encoded_image = request.form[field_name][22:]
    image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    if new_shape:
        image = image.resize(new_shape)
    # np.array(encoded_image) has 4 canals, the last one is not needed
    return np.array(image)[:, :, :-1].transpose((2, 0, 1)) / 255.0


def prepare_result_image(image):
    image = Image.fromarray(
        (image.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
    )
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


@app.route('/apply', methods=['POST'])
def apply():
    image = extract_image(request, 'image')
    reversed_mask = extract_image(request, 'mask', image.shape[1:])
    mask = np.where(reversed_mask, 0, 1).astype(np.float32)
    restored_image = INPAINTER(np.where(mask, image, 0), mask)
    return prepare_result_image(restored_image)


@app.route('/pick_random')
def pick_random():
    images = os.listdir(RANDOM_IMAGES_DIR)
    img_name = random.choice(images)
    return send_file(os.path.join(RANDOM_IMAGES_DIR, img_name))


@app.route('/')
def hello_world():
    return render_template('index.html')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-state-dict')
    parser.add_argument(
        '-r', '--random-images-dir', default=DEFAULT_RANDOM_IMAGES_DIR
    )
    parser.add_argument('-d', '--device', default='cpu')
    args = parser.parse_args()
    setup_app(
        args.model_state_dict, args.random_images_dir, args.device
    ).run()


if __name__ == '__main__':
    main()

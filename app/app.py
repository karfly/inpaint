#!/usr/bin/env python3
import argparse
import base64
import io
import os
import random
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file, session, jsonify

from data import Storage

sys.path.append('..')
from inpaint import make_inpainter

STATIC_DIR = 'static'
RANDOM_IMAGES_DIR = None
DEFAULT_RANDOM_IMAGES_DIR = os.path.join(STATIC_DIR, 'random_images')
INPAINTER = None

app = Flask(__name__)
app.secret_key = 'super secret key 228'


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


def extract_image(image_bytes, new_shape=None):
    image = Image.open(BytesIO(image_bytes))
    if new_shape:
        image = image.resize(new_shape)
    # np.array(encoded_image) has 4 canals, the last one is not needed
    return np.array(image, dtype='uint8')[:, :, :-1].transpose((2, 0, 1))


def prepare_result_image(image):
    image = Image.fromarray((image.transpose((1, 2, 0)) * 255.0).astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pick_random')
def pick_random():
    images = os.listdir(RANDOM_IMAGES_DIR)
    img_name = random.choice(images)
    return send_file(os.path.join(RANDOM_IMAGES_DIR, img_name))


@app.route('/add_image', methods=['POST'])
def add_image():
    image = extract_image(request.files['image'].read(), (256, 256))
    image_id = Storage().save_image(image)
    session['image_id'] = str(image_id)
    return jsonify({
        'image_id': image_id
    })


@app.route('/apply_mask', methods=['POST'])
def apply_mask():
    storage = Storage()

    image_id = session["image_id"]
    step_id = int(request.form['step_id'])

    image = storage.get_image_by_id(image_id).astype(np.float32)
    reversed_mask = extract_image(request.files['mask'].read(), image.shape[1:])
    mask = np.where(reversed_mask, 0, 1).astype(np.float32)
    result = INPAINTER(np.where(mask, image, 0) / 255., mask)

    storage.save_mask_and_result(image_id, step_id, mask, result)
    image_base64 = prepare_result_image(result)
    return jsonify({
        'image_id': image_id,
        'step_id': step_id,
        'result': image_base64
    })


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

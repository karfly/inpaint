#!/usr/bin/env python3

import argparse
import base64
import io
import os
import sys
import random

from PIL import Image
from flask import Flask, render_template, send_file, request

sys.path.append('..')
from inpaint import make_inpainter


STATIC_DIR = 'static'
RANDOM_IMAGES_DIR = None
DEFAULT_RANDOM_IMAGES_DIR = os.path.join(STATIC_DIR, 'random_images')
INPAINTER = None


app = Flask(__name__)


def setup_app(
    model_state_dict=None, random_images_dir=DEFAULT_RANDOM_IMAGES_DIR
):
    global INPAINTER
    global RANDOM_IMAGES_DIR
    INPAINTER = make_inpainter(model_state_dict)
    RANDOM_IMAGES_DIR = random_images_dir
    return app


def extract_image(request, field_name):
    encoded_image = request.form[field_name][22:]
    return Image.open(io.BytesIO(base64.b64decode(encoded_image)))


def prepare_result_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


@app.route('/apply', methods=['POST'])
def apply():
    image = extract_image(request, 'image')
    mask = extract_image(request, 'mask')
    mask = mask.resize(image.size)
    image.paste(mask, (0, 0), mask)
    restored_image = INPAINTER(image, mask)
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
    args = parser.parse_args()
    setup_app(args.model_state_dict, args.random_images_dir).run()


if __name__ == '__main__':
    main()

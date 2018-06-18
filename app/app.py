#!/usr/bin/env python3
import argparse
import os
import random
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file, session, jsonify
from flask_apidoc import ApiDoc

from data import Storage, pil_to_dataURI

sys.path.append('..')
from inpaint import make_inpainter

STATIC_DIR = 'static'
RANDOM_IMAGES_DIR = None
DEFAULT_RANDOM_IMAGES_DIR = os.path.join(STATIC_DIR, 'random_images')
INPAINTER = None

app = Flask(__name__)
app.secret_key = 'super secret key 228'
doc = ApiDoc(app=app)

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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pick_random')
def pick_random():
    """
    @api {get} /pick_random Pick random photo
    @apiVersion 0.0.1
    @apiName pick_random
    @apiGroup Inpaint

    @apiSuccess {File}     File      random image png
    """
    images = os.listdir(RANDOM_IMAGES_DIR)
    img_name = random.choice(images)
    return send_file(os.path.join(RANDOM_IMAGES_DIR, img_name))


@app.route('/add_image', methods=['POST'])
def add_image():
    """
    @api {post} /add_image Add image
    @apiVersion 0.0.1
    @apiName add_image
    @apiGroup Inpaint

    @apiParam {File}       image     Png file 256x256

    @apiSuccess {String}   image_id  Image Id in DB
    """
    image = Image.open(BytesIO(request.files['image'].read()))
    image = image.resize((256, 256))

    image_id = Storage().save_image(image)
    session['image_id'] = str(image_id)
    return jsonify({
        'image_id': image_id
    })


@app.route('/apply_mask', methods=['POST'])
def apply_mask():
    """
    @api {post} /apply_mask Apply mask
    @apiVersion 0.0.1
    @apiName apply_mask
    @apiGroup Inpaint

    @apiParam {String}     image_id  Id of source image to apply
    @apiParam {Integer}    step_id   Stroke number in history
    @apiParam {File}       mask      Png file 256x256, where painted is white and remained is black/transparent

    @apiSuccess {String}   image_id  Id of applied source image (unchanged)
    @apiSuccess {Integer}  step_id   Stroke number in history (unchanged)
    @apiSuccess {String}   result    DataURI of result png
    """
    storage = Storage()

    image_id = request.form['image_id']  # TODO: this is dangerous, but simplier for API
    step_id = int(request.form['step_id'])

    image = storage.get_image_by_id(image_id)
    image = np.array(image, dtype=np.float32)[:, :, :-1].transpose((2, 0, 1))

    reversed_mask = Image.open(BytesIO(request.files['mask'].read()))
    reversed_mask = reversed_mask.resize(image.shape[1:])
    reversed_mask = np.array(reversed_mask, dtype=np.float32)[:, :, :-1].transpose((2, 0, 1))
    mask = np.where(reversed_mask, 0, 1).astype(np.float32)

    result = INPAINTER(np.where(mask, image, 0) / 255., mask)
    result = Image.fromarray((result.transpose((1, 2, 0)) * 255.).astype(np.uint8))

    mask_image = Image.fromarray((mask.transpose((1, 2, 0)) * 255).astype(np.uint8))
    storage.save_mask_and_result(image_id, step_id, mask_image, result)
    return jsonify({
        'image_id': image_id,
        'step_id': step_id,
        'result': pil_to_dataURI(result)
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

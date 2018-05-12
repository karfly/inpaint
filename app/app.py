import base64
import io
import os
import random

from PIL import Image, ImageOps
from flask import Flask, render_template, send_file, request


STATIC_DIR = 'static/images'


app = Flask(__name__)


def extract_image(request, field_name):
    return request.form[field_name][22:]


@app.route('/apply', methods=['POST'])
def apply():
    encoded_image = extract_image(request, 'image')
    encoded_mask = extract_image(request, 'mask')
    image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    mask = Image.open(io.BytesIO(base64.b64decode(encoded_mask)))
    mask = mask.resize(image.size)
    image.paste(mask, (0, 0), mask)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


@app.route('/pick_random')
def pick_random():
    images = os.listdir(STATIC_DIR)
    img_name = random.choice(images)
    return send_file(os.path.join(STATIC_DIR, img_name))


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

import base64
import os
import random
from io import BytesIO
from os.path import join as pj

from PIL import Image, ImageOps
from flask import Flask, render_template, send_file, request

app = Flask(__name__)


@app.route('/apply', methods=['POST'])
def apply():
    encoded_image = request.form['image'][22:]
    encoded_mask = request.form['mask'][22:]
    image = Image.open(BytesIO(base64.b64decode(encoded_image)))
    mask = Image.open(BytesIO(base64.b64decode(encoded_mask)))
    mask = mask.resize(image.size)
    image.paste(mask, (0, 0), mask)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


@app.route('/pick_random')
def pick_random():
    images = os.listdir('./static/images')
    img_name = random.choice(images)
    return send_file(pj('./static/images', img_name))


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

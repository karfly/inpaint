# Deep image inpainting
<img src="https://github.com/karfly/inpaint/blob/master/readme/demo.png" width="1024">

As our final LSML project we decided to create online AI-tool for image correction using **inpainting**. There are a lot of works where this problem is solved with Deep Learning. We took [this fresh paper](https://arxiv.org/abs/1804.07723) by NVidia researches as a base because they archived very spectacular results and can deservedly be considered a state-of-the-art right now.

We thought it would be interesting to apply this tool for face correction, so used [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for learning. But in the original dataset there are a lot of low-quality images, that's why we finally used [CelebA-HQ dataset](https://arxiv.org/abs/1710.10196).

Finally we faced a challenge to create pretty web-page to allow people to use our tool online. You can try it [here](https://github.com/karfly/inpaint).

##### From left to right: original & corrupted & restored
<img src="https://github.com/karfly/inpaint/blob/master/readme/original.png" width="256"> <img src="https://github.com/karfly/inpaint/blob/master/readme/corrupted.png" width="256"> <img src="https://github.com/karfly/inpaint/blob/master/readme/restored.png" width="256">

## Usage
The fastest way to start playing with the demo is (requires [Docker](https://www.docker.com/)):
```
>>> git clone https://github.com/karfly/inpaint
>>> cd inpaint
>>> docker build -t inpaint_image .
>>> ./run.sh
<visit localhost:8003 in your favourite browser>
```

Current limitations:
- photos must have resolution 256x256
- photos must be similar to ones from CelebA-HQ dataset

If you want to explore the project more deeply, here are some notes:
- the project supports only Python 3
- all the dependencies are listed in `requirements.txt`
- a well-documented interface of the main library (with the original model and the loss used when training) is in `inpaint/__init__.py`
- to run the app locally see the example of a command in app/run.sh and the documentation of the function `setup_app` in `app/app.py`

## Contributors
- [Ivan Golovanov](https://github.com/legendawes) - mask generation, research.
- [Yury Gorishniy](https://github.com/StrausMG) - backend, data manipulation, inpaint loss.
- [Vladimir Ivashkin](https://github.com/vlivashkin) - frontend, backend.
- [Karim Iskakov](https://github.com/karfly) - model training, mask generation, CelebA-HQ generation.

## References
- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
- [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)

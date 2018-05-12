# Deep image inpainting

Original             |  Corrupted          |  Restored
:-------------------------:|:-------------------------:|:-------------------------
![](https://github.com/karfly/inpaint/blob/master/readme/original.png)  |\
![](https://github.com/karfly/inpaint/blob/master/readme/corrupted.png) |\
![](https://github.com/karfly/inpaint/blob/master/readme/restored.png)

As our final LSML project we decided to create online AI-tool for image correction using **inpainting**. There are a lot of works where this problem is solved with Deep Learning. We took [this fresh paper](https://arxiv.org/abs/1804.07723) by NVidia researches as a base because they archived very spectacular results and can deservedly be considered a state-of-the-art right now.

We thought it would be interesting to apply this tool for face correction, so used [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for learning. But in the original dataset there are a lot of low-quality images, that's why we finally used [CelebA-HQ dataset](https://arxiv.org/abs/1710.10196).

Finally we faced a challenge to create pretty web-page to allow people to use our tool online. You can try it [here](https://github.com/karfly/inpaint).

**!!SCREENSHOT HERE!!**

## Run


## Contributiors
- [Ivan Golovanov](https://github.com/legendawes) - frontend, backend, mask generation, research.
- [Yury Gorishniy](https://github.com/StrausMG) - backend, data manipulation, inpaint loss.
- [Karim Iskakov](https://github.com/karfly) - model training, mask generation, CelebA-HQ generation.

## References
- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
- [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
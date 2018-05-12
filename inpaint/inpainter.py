import torch

from .module import InpaintModule


class Inpainter:
    def __init__(self, model=None):
        self._model = model

    def __call__(self, img, mask):
        if self._model:
            restored_img, _ = self._model(img, mask)
        else:
            restored_img = img
        return restored_img


def make_inpainter(state_dict_path=None):
    if state_dict_path:
        model = InpaintModule()
        model.load_state_dict(torch.load(state_dict_path))
    else:
        model = None
    return Inpainter(model)

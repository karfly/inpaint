import numpy as np
import torch

from .module import InpaintNet


class Inpainter:
    def __init__(self, model=None):
        self._model = model

    def __call__(self, img, mask):
        if self._model:
            restored_img, _ = self._model(img, mask)
            restored_img = restored_img.to('cpu').data.numpy()
            restored_img = np.where(mask, img, restored_img)
            restored_img = np.clip(restored_img, 0, 1)
        else:
            # makes the holes white, not black
            restored_img = np.where(mask, img, 1.0)
        return restored_img

    def set_device(self, device):
        if self._model:
            self._model.to(device)


def make_inpainter(state_dict_path=None):
    if state_dict_path:
        model = InpaintNet()
        model.load_state_dict(
            torch.load(state_dict_path, map_location=lambda *x: x[0])
        )
        model.eval()
    else:
        model = None
    return Inpainter(model)

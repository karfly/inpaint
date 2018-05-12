import numpy as np
import torch

from .module import InpaintNet


class Inpainter:
    def __init__(self, model=None, device='cpu'):
        self._model = model
        self.set_device(device)

    def __call__(self, img, mask):
        if self._model:
            tensor_img = torch.tensor(
                np.expand_dims(img, axis=0),
                dtype=torch.float32,
                device=self._device
            )
            tensor_mask = torch.tensor(
                np.expand_dims(mask, axis=0),
                dtype=torch.float32,
                device=self._device
            )
            restored_img, _ = self._model(tensor_img, tensor_mask)
            restored_img = restored_img.to('cpu').data.numpy()[0]
            restored_img = np.where(mask, img, restored_img)
            restored_img = np.clip(restored_img, 0, 1)
        else:
            # makes the holes white, not black
            restored_img = np.where(mask, img, 1.0)
        return restored_img

    def set_device(self, device):
        self._device = device
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

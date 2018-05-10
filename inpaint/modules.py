import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


_VGG = None


def _vgg():
    global _VGG
    if _VGG is None:
        _VGG = tv.models.vgg16(True)
    return _VGG


def _perceptual_loss(x_features, y_features):
    return sum(F.l1_loss(x, y) for x, y in zip(x_features, y_features))


def _style_loss(x_gram_matrices, y_gram_matrices, coefs):
    return sum(
        1 / c * F.l1_loss(x, y)
        for x, y, c in zip(x_gram_matrices, y_gram_matrices, coefs)
    )


def _tv_loss(x, mask):
    first = torch.sum(torch.abs(
        x[:, :, :, :-1] * mask[:, :, :, :-1] -
        x[:, :, :, 1: ] * mask[:, :, :, :-1]
    ))
    second = torch.sum(torch.abs(
        x[:, :, :-1, :] * mask[:, :, :-1, :] -
        x[:, :, 1: , :] * mask[:, :, :-1, :]
    ))
    return first + second


class _PretrainedFeaturesGenerator(nn.Module):
    def __init__(self, module, layers, preprocessor=None, reshape=True):
        super().__init__()
        self._module = module
        self._layers = set(layers)
        self._preprocessor = preprocessor or (lambda x: x)
        self._reshape = reshape

    def forward(self, x):
        self._module.eval()
        x = self._preprocessor(x)
        layers = copy.deepcopy(self._layers)
        output = []

        for layer, module in self._module._modules.items():
            if not layers:
                break
            x = module(x)
            if layer in layers:
                output.append(x.view(*x.shape[:2], -1) if self._reshape else x)
                layers.remove(layer)
        return output


def _calculate_gram_matrices(features):
    return [x.matmul(x.transpose(-2, -1)) for x in features]


def _make_normalizer(feature_count, means, stds):
    x = nn.BatchNorm2d(feature_count, affine=False)
    x.weight = nn.Parameter(torch.tensor(means))
    x.bias = nn.Parameter(torch.tensor(stds) ** 2)
    x.weight.requires_grad_(False)
    x.bias.requires_grad_(False)
    return x


class InpaintLoss(nn.Module):
    def __init__(
        self,
        valid_coef=1,
        hole_coef=6,
        perceptual_coef=0.05,
        style_coef=120,
        tv_coef=0.1,
        style_features_generator=None
    ):
        super().__init__()
        self._valid_coef = valid_coef
        self._hole_coef = hole_coef
        self._perceptual_coef = perceptual_coef
        self._style_coef = style_coef
        self._tv_coef = tv_coef
        self._style_features_generator = (
            style_features_generator or
            _PretrainedFeaturesGenerator(
                _VGG.features,
                ('4', '9', '16'),
                # see https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models
                # for undestanding why these figures
                _make_normalizer(
                    3,
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            )
        )

    def forward(self, out, mask, gt):
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)

        loss = loss + self._valid_coef * F.l1_loss(mask * out, mask * gt)
        reversed_mask = 1 - mask
        loss = loss + self._hole_coef * (
            F.l1_loss(reversed_mask * out, reversed_mask * gt)
        )

        out_features = self._style_features_generator(out)
        gt_features = self._style_features_generator(gt)
        loss = loss + self._perceptual_coef * (
            _perceptual_loss(out_features, gt_features)
        )

        out_gram_matrices = _calculate_gram_matrices(out_features)
        gt_gram_matrices = _calculate_gram_matrices(gt_features)
        coefs = [x.shape[-2] * x.shape[-1] for x in out_features]
        loss = loss + self._style_coef * (
            _style_loss(out_gram_matrices, gt_gram_matrices, coefs)
        )

        comp = torch.where(mask.type(torch.uint8), gt, out)
        comp_features = self._style_features_generator(comp)
        loss = loss + self._perceptual_coef * (
            _perceptual_loss(comp_features, gt_features)
        )

        comp_gram_matrices = _calculate_gram_matrices(comp_features)
        loss = loss + self._style_coef * (
            _style_loss(comp_gram_matrices, gt_gram_matrices, coefs)
        )

        loss = loss + self._tv_coef * _tv_loss(comp, mask)

        return loss

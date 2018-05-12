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


class _Normalization(nn.Module):
    def __init__(self, mean, std):
        super(_Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
            requires_grad=False
        )
        self.std = nn.Parameter(
            torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
            requires_grad=False
        )

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def _calculate_gram_matrices(features):
    return [x.matmul(x.transpose(-2, -1)) for x in features]


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
                _vgg().features,
                ('4', '9', '16'),
                # see https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models
                # for undestanding why these figures
                _Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )
        )

    def forward(self, out, mask, gt):
        loss = torch.tensor(
            0, dtype=torch.float32, device=out.device, requires_grad=True
        )

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

        comp = torch.where(mask.type(torch.uint8), gt, out).to(out.device)
        comp_features = self._style_features_generator(comp)
        loss = loss + self._perceptual_coef * (
            _perceptual_loss(comp_features, gt_features)
        )

        comp_gram_matrices = _calculate_gram_matrices(comp_features)
        loss = loss + self._style_coef * (
            _style_loss(comp_gram_matrices, gt_gram_matrices, coefs)
        )

        loss = loss + self._tv_coef * _tv_loss(comp, reversed_mask)

        return loss


class _PartialConv2d(nn.Module):
    """Partial convolution layer (https://arxiv.org/abs/1804.07723)."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.conv_bias = nn.Parameter(
            torch.zeros(out_channels), requires_grad=True
        )

        self.sum_conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.sum_conv.weight.data.fill_(1)
        self.sum_conv.weight.requires_grad_(False)

    def forward(self, x, mask):
        """Do forward pass.

        Parameters
        ----------
        x : FloatTensor
            input feature tensor of shape (b, c, h, w)

        mask : FloatTensor
            binary mask tensor of shape (b, c, h, w)

        Returns
        -------
        x : FloatTensor
        updated_mask : FloatTensor
        """
        assert x.shape == mask.shape, 'x and mask shapes must be equal'

        x_masked = x * mask
        x_after_conv = self.conv(x_masked)

        x_after_conv_normed = x_after_conv  # no norm
        # x_after_conv_normed = torch.where(
        #     mask_norm != 0,
        #     x_after_conv / mask_norm,
        #     torch.zeros_like(x_after_conv)
        # )
        x_after_conv_normed += self.conv_bias.view(1, -1, 1, 1)

        updated_mask_single = (self.sum_conv(mask) > 0).type(torch.float32)
        updated_mask = torch.cat(
            [updated_mask_single] * self.out_channels, dim=1
        )
        updated_mask.to(mask.device)

        return x_after_conv_normed, updated_mask


class _InpaintDownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=2,
        padding='same',
        bn=True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self.padding = padding

        self.bn = bn

        self.pconv = _PartialConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)

        return x, mask


class _InpaintUpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        in_channels_bridge,
        out_channels,
        kernel_size,
        padding='same',
        bn=True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_channels_bridge = in_channels_bridge
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self.padding = padding

        self.bn = bn
        # TODO: align corners!
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.pconv = _PartialConv2d(
            in_channels + in_channels_bridge,
            out_channels,
            kernel_size,
            padding=padding
        )
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, mask, x_bridge, mask_bridge):
        x, mask = self.upsample(x), self.upsample(mask)
        x = torch.cat([x, x_bridge], dim=1)
        mask = torch.cat([mask, mask_bridge], dim=1)

        x, mask = self.pconv(x, mask)

        if self.bn:
            x = self.bn(x)
        x = self.leaky_relu(x)

        return x, mask


class InpaintNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # down
        self.down_blocks = nn.ModuleList([
            _InpaintDownBlock(
                in_channels, 64, 7, stride=2, padding='same', bn=False
            ),
            _InpaintDownBlock(64, 128, 5, stride=2, padding='same', bn=False),
            _InpaintDownBlock(128, 256, 5, stride=2, padding='same', bn=False),
            _InpaintDownBlock(256, 512, 3, stride=2, padding='same', bn=False),
            _InpaintDownBlock(512, 512, 3, stride=2, padding='same', bn=False),
            _InpaintDownBlock(512, 512, 3, stride=2, padding='same', bn=False),
            _InpaintDownBlock(512, 512, 3, stride=2, padding='same', bn=False),
        ])
        self.depth = len(self.down_blocks)

        # up
        self.up_blocks = nn.ModuleList([
            _InpaintUpBlock(512, 512, 512, 3, padding='same', bn=False),
            _InpaintUpBlock(512, 512, 512, 3, padding='same', bn=False),
            _InpaintUpBlock(512, 512, 512, 3, padding='same', bn=False),
            _InpaintUpBlock(512, 256, 256, 3, padding='same', bn=False),
            _InpaintUpBlock(256, 128, 128, 3, padding='same', bn=False),
            _InpaintUpBlock(128, 64, 64, 3, padding='same', bn=False),
            _InpaintUpBlock(64, 3, 3, 3, padding='same', bn=False)
        ])

    def forward(self, x, mask):
        x_bridges, mask_bridges = [], []
        for i in range(self.depth):
            x_bridges.append(x)
            mask_bridges.append(mask)
            x, mask = self.down_blocks[i](x, mask)

        for i in range(self.depth):
            x, mask = self.up_blocks[i](
                x, mask, x_bridges[-i - 1], mask_bridges[-i - 1]
            )

        return x, mask

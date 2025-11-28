import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm as snorm
from torchvision.models import VGG19_Weights, vgg19

import config

logger = config.create_logger("INFO", __file__)


def _init_scaled_weights(module: nn.Module, scale: float):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight.data, a=0.0, mode="fan_in")
        module.weight.data *= scale

        if module.bias is not None:
            init.constant_(module.bias.data, 0.0)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        activation: Literal["leaky_relu", "tanh"] | None = None,
        spectral_norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential()

        if spectral_norm:
            self.conv_block.append(
                snorm(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    )
                )
            )
        else:
            self.conv_block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )

        if activation:
            match activation.lower():
                case "leaky_relu":
                    self.conv_block.append(
                        nn.LeakyReLU(
                            negative_slope=config.LEAKY_RELU_NEGATIVE_SLOPE_VALUE,
                            inplace=True,
                        )
                    )
                case "tanh":
                    self.conv_block.append(nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class SubPixelConvBlock(nn.Module):
    def __init__(
        self,
        channels_count: int,
        kernel_size: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.subpixel_conv_block = nn.Sequential(
            ConvBlock(
                in_channels=channels_count,
                out_channels=channels_count * (scaling_factor**2),
                kernel_size=kernel_size,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.subpixel_conv_block(x)


class ResDenseBlock(nn.Module):
    def __init__(
        self,
        channels_count: int,
        growth_channels_count: int,
        kernel_size: int,
        conv_layers_count: int,
    ) -> None:
        super().__init__()

        self.conv_layers_count: int = conv_layers_count
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(
            ConvBlock(
                in_channels=channels_count,
                out_channels=growth_channels_count,
                kernel_size=kernel_size,
                activation="leaky_relu",
            )
        )

        for i in range(1, conv_layers_count - 1):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=channels_count + i * growth_channels_count,
                    out_channels=growth_channels_count,
                    kernel_size=kernel_size,
                    activation="leaky_relu",
                )
            )

        self.conv_layers.append(
            ConvBlock(
                in_channels=channels_count
                + (conv_layers_count - 1) * growth_channels_count,
                out_channels=channels_count,
                kernel_size=kernel_size,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs = [x]
        outputs.append(self.conv_layers[0](x))

        for i in range(1, self.conv_layers_count):
            outputs.append(self.conv_layers[i](torch.cat(outputs, 1)))

        return outputs[-1] * config.RESIDUAL_SCALING_VALUE + x


class RRDB(nn.Module):
    def __init__(
        self,
        channels_count: int,
        growth_channels_count: int,
        kernel_size: int,
        conv_layers_count: int,
        res_dense_blocks_count: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            *[
                ResDenseBlock(
                    channels_count=channels_count,
                    growth_channels_count=growth_channels_count,
                    kernel_size=kernel_size,
                    conv_layers_count=conv_layers_count,
                )
                for _ in range(res_dense_blocks_count)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) * config.RESIDUAL_SCALING_VALUE + x


class Generator(nn.Module):
    def __init__(
        self,
        channels_count: int,
        growth_channels_count: int,
        large_kernel_size: int,
        small_kernel_size: int,
        res_dense_blocks_count: int,
        rrdb_count: int,
        scaling_factor: Literal[2, 4, 8],
    ) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=channels_count,
            kernel_size=large_kernel_size,
            padding=4,
        )

        self.rrdb = nn.Sequential(
            *[
                RRDB(
                    channels_count=channels_count,
                    growth_channels_count=growth_channels_count,
                    kernel_size=small_kernel_size,
                    conv_layers_count=5,
                    res_dense_blocks_count=res_dense_blocks_count,
                )
                for _ in range(rrdb_count)
            ]
        )

        self.conv_block2 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count,
            kernel_size=small_kernel_size,
        )

        self.subpixel_conv_blocks = nn.Sequential(
            *[
                SubPixelConvBlock(
                    channels_count=channels_count,
                    kernel_size=small_kernel_size,
                    scaling_factor=2,
                )
                for _ in range(int(math.log2(scaling_factor)))
            ]
        )

        self.conv_block3 = ConvBlock(
            in_channels=channels_count,
            out_channels=3,
            kernel_size=large_kernel_size,
            padding=4,
            activation="tanh",
        )

        self.apply(
            lambda fn: _init_scaled_weights(fn, scale=config.WEIGHTS_SCALING_VALUE)
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv_block1(x)
        residual = output
        output = self.rrdb(output)
        output = self.conv_block2(output)
        output += residual
        output = self.subpixel_conv_blocks(output)
        output = self.conv_block3(output)

        return output


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_count: int,
    ) -> None:
        super().__init__()

        self.conv0 = ConvBlock(
            in_channels=in_channels,
            out_channels=channels_count,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="leaky_relu",
            spectral_norm=False,
            bias=True,
        )

        self.conv1 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv2 = ConvBlock(
            in_channels=channels_count * 2,
            out_channels=channels_count * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv3 = ConvBlock(
            in_channels=channels_count * 4,
            out_channels=channels_count * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv4 = ConvBlock(
            in_channels=channels_count * 8,
            out_channels=channels_count * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv5 = ConvBlock(
            in_channels=channels_count * 4,
            out_channels=channels_count * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv6 = ConvBlock(
            in_channels=channels_count * 2,
            out_channels=channels_count,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv7 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv8 = ConvBlock(
            in_channels=channels_count,
            out_channels=channels_count,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="leaky_relu",
            spectral_norm=True,
            bias=False,
        )

        self.conv9 = ConvBlock(
            in_channels=channels_count,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=None,
            spectral_norm=False,
            bias=True,
        )

        self.apply(lambda fn: _init_scaled_weights(fn, scale=1.0))

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        x3 = F.interpolate(
            self.conv3(x2), scale_factor=2, mode="bilinear", align_corners=False
        )

        x4 = F.interpolate(
            self.conv4(x3) + x2, scale_factor=2, mode="bilinear", align_corners=False
        )

        x5 = F.interpolate(
            self.conv5(x4) + x1, scale_factor=2, mode="bilinear", align_corners=False
        )

        output = self.conv7(self.conv6(x5) + x0)
        output = self.conv8(output)
        output = self.conv9(output)

        return output


class TruncatedVGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers: dict = {
            2: "conv1_2",
            7: "conv2_2",
            16: "conv3_4",
            25: "conv4_4",
            34: "conv5_4",
        }

        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)

        self.features = vgg19_model.features[:35]

        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        output = {}

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i in self.layers:
                output[self.layers[i]] = x.clone()

        return output

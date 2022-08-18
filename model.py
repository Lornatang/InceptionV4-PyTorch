# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "InceptionV4", "InceptionV4ResNetV2",
    "BasicConv2d", "ReductionA",
    "InceptionV4Stem", "InceptionA", "InceptionResNetB", "ReductionResNetB", "InceptionC",
    "InceptionV4ResNetStem", "InceptionResNetA", "InceptionResNetB", "ReductionResNetB", "InceptionResNetC",
    "inception_v4",
]


class InceptionV4(nn.Module):

    def __init__(
            self,
            k: int = 192,
            l: int = 224,
            m: int = 256,
            n: int = 384,
            num_classes: int = 1000,
    ) -> None:
        super(InceptionV4, self).__init__()
        self.features = nn.Sequential(
            InceptionV4Stem(3),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            ReductionA(384, k, l, m, n),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            ReductionB(1024),
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536),
        )

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(1536, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.global_average_pooling(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class InceptionV4ResNetV2(nn.Module):

    def __init__(
            self,
            k: int = 192,
            l: int = 224,
            m: int = 256,
            n: int = 384,
            num_classes: int = 1000,
    ) -> None:
        super(InceptionV4ResNetV2, self).__init__()
        self.features = nn.Sequential(
            InceptionV4ResNetStem(3),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            InceptionResNetA(320, 0.17),
            ReductionA(320, k, l, m, n),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            InceptionResNetB(1088, 0.10),
            ReductionResNetB(1088),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, True),
            InceptionResNetC(2080, 0.20, False),
        )

        self.conv = BasicConv2d(2080, 1536, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(1536, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.conv(out)
        out = self.global_average_pooling(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class InceptionV4Stem(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionV4Stem, self).__init__()
        self.conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.mixed_3a_branch_0 = nn.MaxPool2d((3, 3), (2, 2))
        self.mixed_3a_branch_1 = BasicConv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.mixed_4a_branch_0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        )

        self.mixed_5a_branch_0 = BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.mixed_5a_branch_1 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2d_1a_3x3(x)
        out = self.conv2d_2a_3x3(out)
        out = self.conv2d_2b_3x3(out)

        mixed_3a_branch_0 = self.mixed_3a_branch_0(out)
        mixed_3a_branch_1 = self.mixed_3a_branch_1(out)
        mixed_3a_out = torch.cat([mixed_3a_branch_0, mixed_3a_branch_1], 1)

        mixed_4a_branch_0 = self.mixed_4a_branch_0(mixed_3a_out)
        mixed_4a_branch_1 = self.mixed_4a_branch_1(mixed_3a_out)
        mixed_4a_out = torch.cat([mixed_4a_branch_0, mixed_4a_branch_1], 1)

        mixed_5a_branch_0 = self.mixed_5a_branch_0(mixed_4a_out)
        mixed_5a_branch_1 = self.mixed_5a_branch_1(mixed_4a_out)
        mixed_5a_out = torch.cat([mixed_5a_branch_0, mixed_5a_branch_1], 1)

        return mixed_5a_out


class InceptionV4ResNetStem(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionV4ResNetStem, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            BasicConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((3, 3), (2, 2)),
            BasicConv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d((3, 3), (2, 2)),
        )
        self.branch_0 = BasicConv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        features = self.features(x)
        branch_0 = self.branch_0(features)
        branch_1 = self.branch_1(features)
        branch_2 = self.branch_2(features)
        branch_3 = self.branch_3(features)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

        return out


class InceptionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        brance_3 = self.brance_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, brance_3], 1)

        return out


class InceptionResNetA(nn.Module):
    def __init__(
            self,
            in_channels: int,
            scale_factor: float,
    ) -> None:
        super(InceptionResNetA, self).__init__()
        self.scale_factor = scale_factor
        self.branch_0 = BasicConv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.conv = nn.Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = self.conv(torch.cat([branch_0, branch_1, branch_2], 1))

        out = torch.mul(out, self.scale_factor)
        out = torch.add(out, x)
        out = self.relu(out)

        return out


class ReductionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
            k: int,
            l: int,
            m: int,
            n: int,
    ) -> None:
        super(ReductionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, n, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(k, l, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(l, m, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = torch.cat([branch_0, branch_1, branch_2], 1)

        return out


class InceptionB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionB, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

        return out


class InceptionResNetB(nn.Module):
    def __init__(
            self,
            in_channels: int,
            scale_factor: float,
    ) -> None:
        super(InceptionResNetB, self).__init__()
        self.scale_factor = scale_factor

        self.branch_0 = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
        )
        self.conv = nn.Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)

        out = self.conv(torch.cat([branch_0, branch_1], 1))

        out = torch.mul(out, self.scale_factor)
        out = torch.add(out, x)
        out = self.relu(out)

        return out


class ReductionB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(ReductionB, self).__init__()
        self.branch_0 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = torch.cat([branch_0, branch_1, branch_2], 1)

        return out


class ReductionResNetB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(ReductionResNetB, self).__init__()
        self.branch_0 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(256, 384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        )
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(256, 288, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(256, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(288, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        )
        self.branch_3 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

        return out


class InceptionC(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionC, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch_1 = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1_1 = BasicConv2d(384, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_1_2 = BasicConv2d(384, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(384, 448, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            BasicConv2d(448, 512, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        )
        self.branch_2_1 = BasicConv2d(512, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_2_2 = BasicConv2d(512, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1)),
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)

        branch_1_1 = self.branch_1_1(branch_1)
        branch_1_2 = self.branch_1_2(branch_1)
        x1 = torch.cat([branch_1_1, branch_1_2], 1)

        branch_2 = self.branch_2(x)
        branch_2_1 = self.branch_2_1(branch_2)
        branch_2_2 = self.branch_2_2(branch_2)
        x2 = torch.cat([branch_2_1, branch_2_2], 1)

        x3 = self.branch_3(x)

        out = torch.cat([branch_0, x1, x2, x3], 1)

        return out


class InceptionResNetC(nn.Module):
    def __init__(
            self,
            in_channels: int,
            scale_factor: float,
            activation: bool,
    ) -> None:
        super(InceptionResNetC, self).__init__()
        self.scale_factor = scale_factor
        self.activation = activation

        self.branch_0 = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv = nn.Conv2d(448, 2080, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)

        out = self.conv(torch.cat([branch_0, branch_1], 1))

        out = torch.mul(out, self.scale_factor)
        out = torch.add(out, x)

        if self.activation:
            out = self.relu(out)

        return out


def inception_v4(**kwargs: Any) -> InceptionV4:
    model = InceptionV4(k=192, l=224, m=256, n=384, **kwargs)

    return model


def inception_v4_resnet_v2(**kwargs: Any) -> InceptionV4:
    model = InceptionV4ResNetV2(k=256, l=256, m=384, n=384, **kwargs)

    return model

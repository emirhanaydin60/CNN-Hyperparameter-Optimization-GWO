import torch
import torch.nn as nn


def _same_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) // 2) * dilation


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=_same_padding(kernel_size, dilation),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        padding = _same_padding(kernel_size, dilation)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class InceptionLiteBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        branch_sizes = [out_channels // 3, out_channels // 3, out_channels - 2 * (out_channels // 3)]
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_sizes[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_sizes[0]),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_sizes[1],
                kernel_size=3,
                padding=_same_padding(3, dilation),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(branch_sizes[1]),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_sizes[2],
                kernel_size=5,
                padding=_same_padding(5, dilation),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(branch_sizes[2]),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.relu(self.bn(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()
        hidden_channels = max(1, channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.pool(x)
        weights = self.fc(weights).view(weights.size(0), weights.size(1), 1, 1)
        return x * weights


class HybridCNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        img_size=32,
        kernel_size=3,
        base_filters=32,
        dilation=1,
        final_neurons=256,
        dropout=0.25,
        se_ratio=8,
        num_classes=10,
    ):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, base_filters, kernel_size=kernel_size, dilation=dilation)
        self.residual1 = ResidualBlock(base_filters, kernel_size=kernel_size, dilation=dilation)
        self.inception = InceptionLiteBlock(base_filters, base_filters, dilation=dilation)
        self.residual2 = ResidualBlock(base_filters, kernel_size=kernel_size, dilation=dilation)
        self.se = SEBlock(base_filters, reduction_ratio=se_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters, final_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_neurons, num_classes),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            _ = self.forward_features(dummy)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.residual1(x)
        x = self.inception(x)
        x = self.residual2(x)
        x = self.se(x)
        return self.pool(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

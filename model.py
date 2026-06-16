import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, img_size=28, filter_size=3, filters=16, dilation=1, final_neurons=128, dropout=0.25, num_classes=10):
        super().__init__()
        padding = ((filter_size - 1) // 2) * dilation
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=filter_size, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters * 2, kernel_size=filter_size, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # compute flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            feat = self.features(dummy)
            flattened = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, final_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_neurons, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

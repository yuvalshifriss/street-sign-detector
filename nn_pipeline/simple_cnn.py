# nn_pipeline/model/simple_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for bounding box regression.
    Outputs a vector of length `num_outputs` (e.g., [x, y, w, h]).
    """
    def __init__(self, num_outputs: int = 4):
        """
        Args:
            num_outputs (int): Number of output values. Default is 4 for bounding box regression.
        """
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6x6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, 48, 48)

        Returns:
            torch.Tensor: Output tensor of shape (B, num_outputs)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

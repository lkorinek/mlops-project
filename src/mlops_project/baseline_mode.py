import torch
from torch import nn
import torch.nn.functional as F

class PneumoniaClassifier(nn.Module):
    """Baseline CNN for Pneumonia classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)  # Adjust for input size after pooling
        self.fc2 = nn.Linear(128, 2)  # Binary classification (Pneumonia vs Normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = PneumoniaClassifier()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Example input (3 channels, 150x150 resized image)
    dummy_input = torch.randn(1, 3, 150, 150)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFromDiagram(nn.Module):
    def __init__(self, input_channels=200, num_classes=16, patch_size=16):
        super(CNNFromDiagram, self).__init__()

        # Conv1: input_channels -> 100, kernel 3x3
        # Dodano padding=1 żeby działało z mniejszymi patch_size (np. 8)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=100, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Conv2: 100 -> 100, kernel 3x3
        # Dodano padding=1 żeby działało z mniejszymi patch_size
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Oblicz rozmiar wyjścia po konwolucjach - dynamicznie jak w train.ipynb
        dummy_input = torch.zeros(1, input_channels, patch_size, patch_size)
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        flatten_dim = x.view(1, -1).shape[1]

        # FC layers
        self.fc1 = nn.Linear(flatten_dim, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def extract_features(self, x):
        """
        Ekstrahuje embedding (cechy) zamiast klasyfikacji
        Zwraca: (batch, 84) - embedding przed klasyfikatorem
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        features = F.relu(self.fc1(x))  # (batch, 84)
        return features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

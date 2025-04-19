# yoga_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class YogaCNN(nn.Module):
    def __init__(self, num_classes=6, rating_classes=3):
        super(YogaCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # [B, 32, 64, 64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # [B, 64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # [B, 64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # [B, 128, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # [B, 128, 16, 16]
        )

        self.flatten = nn.Flatten()

        self.fc_asana = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)   # for asana class
        )

        self.fc_rating = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, rating_classes)  # for good/avg/poor
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        asana_output = self.fc_asana(x)
        rating_output = self.fc_rating(x)
        return asana_output, rating_output

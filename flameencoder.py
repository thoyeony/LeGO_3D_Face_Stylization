import torch
import torch.nn as nn

class FlameShapeCodeEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(FlameShapeCodeEncoder, self).__init__()
        
        # Define layers for the encoder
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        latent_code = self.fc3(x)
        return latent_code

class FlameExpressionCodeEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(FlameExpressionCodeEncoder, self).__init__()
        
        # Define layers for the encoder
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        latent_code = self.fc3(x)
        return latent_code

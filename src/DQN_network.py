import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                device = "cpu"
                ):
        super(DQN, self).__init__()

        self.device = device
        self.to(self.device)

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        ).to(self.device)


    def forward(self, x):
        # Ensure x is a tensor and is on the correct device as during test
        # the state passed is not transofmed in a tensor/batch of tensors

        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = torch.nn.functional.one_hot(x, num_classes = self.input_dim) # num_classes should match the input dimension of your first linear layer in self.model
        x = x.float()
        x = x.to(self.device)

        return self.model(x)
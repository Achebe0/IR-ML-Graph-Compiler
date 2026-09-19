import torch
import torch.nn as nn
import json


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))


import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
import json
import pickle


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))


## recording the operations without execution
def torch_export_ffn(d_model=512, d_ff=2048, batch_size=1):
    model = FeedForwardNetwork(d_model= d_model, d_ff=d_ff).eval()
    example_input = torch.randn(batch_size, d_model)
    return torch.export.export(model, (example_input,))






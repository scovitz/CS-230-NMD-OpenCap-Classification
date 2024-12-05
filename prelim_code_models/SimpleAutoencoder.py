import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import nimblephysics as nimble
import logging

ACTIVATION_FUNCS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}



class SimpleAutoencoder(nn.Module):
    raw_dim: int
    compressed_dim: int
    in_mlp_hidden_dims: List[int]
    out_mlp_hidden_dims: List[int]

    def __init__(self,
                 raw_dim: int,
                 in_mlp_hidden_dims: List[int] = [512],
                 compressed_dim: int = 1024,
                 out_mlp_hidden_dims: List[int] = [512],
                 dropout: bool = False,
                 dropout_prob: float = 0.0,
                 batchnorm: bool = False,
                 activation: str = 'sigmoid',
                 device: str = 'cpu'):
        super(SimpleAutoencoder, self).__init__()

        self.raw_dim = raw_dim
        self.compressed_dim = compressed_dim
        self.in_mlp_hidden_dims = in_mlp_hidden_dims
        self.out_mlp_hidden_dims = out_mlp_hidden_dims

        self.in_mlp = []
        dims = [raw_dim] + in_mlp_hidden_dims + [compressed_dim]
        for i, (h0, h1) in enumerate(zip(dims[:-1], dims[1:])):
            if dropout:
                self.in_mlp.append(nn.Dropout(dropout_prob))
            if batchnorm:
                self.in_mlp.append(nn.BatchNorm1d(h0))
            self.in_mlp.append(nn.Linear(h0, h1, dtype=torch.float32, device=device))
            if i < len(dims) - 2:
                self.in_mlp.append(ACTIVATION_FUNCS[activation])
        self.in_mlp = nn.Sequential(*self.in_mlp)

        self.out_mlp = []
        dims = [compressed_dim] + out_mlp_hidden_dims + [raw_dim]
        out_mlp = []
        for i, (h0, h1) in enumerate(zip(dims[:-1], dims[1:])):
            if dropout:
                out_mlp.append(nn.Dropout(dropout_prob))
            if batchnorm:
                out_mlp.append(nn.BatchNorm1d(h0))
            out_mlp.append(nn.Linear(h0, h1, dtype=torch.float32, device=device))
            if i < len(dims) - 2:
                out_mlp.append(ACTIVATION_FUNCS[activation])
        self.out_mlp = nn.Sequential(*out_mlp)

        logging.info(f"{self.in_mlp=}")
        logging.info(f"{self.out_mlp=}")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Process each point independently
        encoded = self.in_mlp(tensor)
        # Return the outputs
        decoded = self.out_mlp(encoded)
        return decoded
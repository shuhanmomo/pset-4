"""
Implement a neural field using a simple 2D grid.
"""

import torch
import torch.nn as nn
from typing import Tuple
import jaxtyping


class GRID(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        side_length: int,  # image length/height
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.side_length = side_length
        self.grid = torch.nn.Parameter(
            torch.randn(out_features, self.side_length, self.side_length)
        )  # a 2D Grid

    def forward(self, coords: jaxtyping.Float[torch.Tensor, "N D"]) -> Tuple[
        jaxtyping.Float[torch.Tensor, "N out_features"],
        jaxtyping.Float[torch.Tensor, "N D"],
    ]:

        batch = coords.size(0)
        coords = coords.requires_grad_(True)
        grid_pts = coords.view(1, 1, batch, 2)
        sampled_values = torch.nn.functional.grid_sample(
            input=self.grid.unsqueeze(0),
            grid=grid_pts,
            mode="bilinear",
            align_corners=False,
        )
        sampled_values = (
            sampled_values.transpose(1, -2)
            .transpose(-1, -2)
            .reshape(batch, self.out_features)
        )

        return sampled_values, coords

import torch
from math import sin, cos


class PositionalEncoder:
    def __init__(self, d_model: int, scalar: int = 10000) -> None:
        self.d_model = d_model
        self.scalar = scalar

    def __call__(self, pos: int) -> torch.tensor:
        pos_emb = torch.zeros(self.d_model)
        for i in range(self.d_model // 2):
            exp = 2 * i / self.d_model
            pos_emb[2 * i] = sin(pos / self.scalar**exp)
            pos_emb[2 * i + 1] = cos(pos / self.scalar**exp)
        return pos_emb

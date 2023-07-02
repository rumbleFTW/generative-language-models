import torch
from math import sin, cos


class PositionalEncoder:
    def __init__(self, embedding_dim: int, scalar: int = 10000) -> None:
        self.embedding_dim = embedding_dim
        self.scalar = scalar

    def __call__(self, pos: int) -> torch.tensor:
        pos_emb = torch.zeros(self.embedding_dim)
        for i in range(self.embedding_dim // 2):
            exp = 2 * i / self.embedding_dim
            pos_emb[2 * i] = sin(pos / self.scalar**exp)
            pos_emb[2 * i + 1] = cos(pos / self.scalar**exp)
        return pos_emb

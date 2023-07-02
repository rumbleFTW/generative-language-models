from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from tqdm import tqdm


class SkipGramNet(nn.Module):
    """
    [Model]
    Neural network that learns the embedding matrix.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(SkipGramNet, self).__init__()
        self.inp = nn.Linear(vocab_size, embedding_dim, dtype=torch.float32)
        self.emb = nn.Linear(embedding_dim, vocab_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        o1 = self.inp(x)
        o2 = self.emb(o1)
        return nn.functional.log_softmax(o2, dim=0)


class SkipGram:
    """
    [Controller]
    Skipgram implementation with SkipGramNet handling APIs.
    """

    def __init__(self, embedding_dim: int, context_size: int) -> None:
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.net = None

    def fit(self, corpus: List[str], epochs: int, device: str = "cpu"):
        self.vocab = list(set(corpus))
        self.vocab_size = len(self.vocab)
        self.index_table = {string: idx for idx, string in enumerate(self.vocab)}
        self.net = SkipGramNet(self.vocab_size, self.embedding_dim)

        X_train = []
        y_train = []

        corpus_seq = [self.index_table[string] for string in corpus]

        for idx, target in enumerate(corpus_seq):
            contexts = corpus_seq[
                max(0, idx - self.context_size) : idx + self.context_size + 1
            ]
            for context in contexts:
                X = [0] * self.vocab_size
                X[target] = 1
                y = [0] * self.vocab_size
                y[context] = 1
                X_train.append(X)
                y_train.append(y)

        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
            y_train, dtype=torch.float32
        )

        X_train = X_train.to(device)
        y_train = y_train.to(device)

        self.net = self.net.to(device)

        optimizer = optim.Adam(self.net.parameters())
        t = tqdm(range(epochs))
        for _ in t:
            train_loss = 0.0
            self.net.train()
            for X, y in zip(X_train, y_train):
                y_hat = self.net(X)
                y_hat = y_hat.squeeze(0)
                y = y.long()
                loss = torch.nn.functional.nll_loss(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                t.set_description(f"Train loss: {loss:.3f}")
            train_loss /= len(X_train)

        embedding_matrix = self.net.state_dict()["emb.weight"].data

        self.embedding_table = {
            string: tuple(embedding_matrix[self.index_table[string]].tolist())
            for string in self.vocab
        }

        self.embedding_table_inv = {
            tuple(embedding_matrix[self.index_table[token]].tolist()): token
            for token in self.vocab
        }

    def save(self, path: str):
        with open(path, "w") as table:
            json.dump(self.embedding_table, table, indent=4)

    def load(self, path: str):
        with open(path, "r") as table:
            self.embedding_table = {
                key: tuple(value) for key, value in json.load(table).items()
            }
            self.embedding_table_inv = {
                tuple(value): key for key, value in self.embedding_table.items()
            }
            self.vocab_size = len(self.embedding_table)

    def __call__(self, string: str) -> torch.tensor:
        return torch.tensor(self.embedding_table[string])

    def encode(self, sequence: List[str]) -> tuple:
        return torch.tensor([self.embedding_table[string] for string in sequence])

    def decode(self, sequence: List[str]) -> tuple:
        return [self.embedding_table_inv[tuple(token.tolist())] for token in sequence]

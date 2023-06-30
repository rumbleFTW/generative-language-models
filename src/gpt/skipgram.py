import torch
import torch.nn as nn
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
        self.inp = nn.Linear(vocab_size, embedding_dim)
        self.emb = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        o1 = self.inp(x)
        o2 = self.emb(o1)
        return nn.functional.softmax(o2, dim=0)


class SkipGram:
    """
    [Controller]
    Skipgram implementation with SkipGramNet handling APIs.
    """

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        corpus: str = None,
        type="word_level",
    ) -> None:
        if corpus:
            if type == "word_level":
                self.vocab = list(set(corpus.split(" ")))
            elif type == "char_level":
                self.vocab = list(set(list(corpus)))
            self.vocab_size = len(self.vocab)
            self.index_table = {string: idx for idx, string in enumerate(self.vocab)}
        self.type = type
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.net = None

    def fit(self, corpus: str, epochs: int, device: str = "cpu"):
        self.net = SkipGramNet(self.vocab_size, self.embedding_dim)

        X_train = []
        y_train = []

        if self.type == "word_level":
            corpus_seq = [self.index_table[string] for string in corpus.split(" ")]
        elif self.type == "char_level":
            corpus_seq = [self.index_table[string] for string in list(corpus)]

        for idx, target in tqdm(enumerate(corpus_seq)):
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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        for _ in tqdm(range(epochs)):
            for X, y in zip(X_train, y_train):
                y_hat = self.net(X)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        embedding_matrix = self.net.state_dict()["emb.weight"].data

        self.embedding_table = {
            token: tuple(
                float(x) for x in embedding_matrix[self.index_table[token]].tolist()
            )
            for token in self.vocab
        }

        self.embedding_table_inv = {
            tuple(
                float(x) for x in embedding_matrix[self.index_table[token]].tolist()
            ): token
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
            self.vocab_size = len(self.index_table)

    def encode(self, sequence: iter):
        if self.type == "word_level":
            return [self.embedding_table[string] for string in sequence.split(" ")]
        elif self.type == "char_level":
            return [self.embedding_table[string] for string in list(sequence)]

    def decode(self, sequence: iter):
        return [self.embedding_table_inv[token] for token in sequence]

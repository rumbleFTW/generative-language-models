import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import split


class NGramNet(nn.Module):
    """
    [Model]
    A simple neural network with 3 layers: Embedding, a bi-directional LSTM & an output fully-connected layer.
    """

    def __init__(self, vocab_size: int) -> None:
        super(NGramNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.lstm = nn.LSTM(vocab_size, vocab_size, bidirectional=True)
        self.fc = nn.Linear(vocab_size * 2, vocab_size)

    def forward(self, idx: int):
        emb = self.embedding(idx)
        lstm, _ = self.lstm(emb)
        return self.fc(lstm[-1:,])


class NGram:
    """
    [Controller]
    NGram controller with NGramNet handling APIs.
    """

    def __init__(self, vocab_size: int) -> None:
        self.net = NGramNet(vocab_size)

    def fit(
        self,
        X_data: torch.tensor,
        y_data: torch.tensor,
        epochs: int,
        checkpt_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        optimizer = optim.Adam(self.net.parameters())
        t = tqdm(range(epochs))
        self.net = self.net.to(device)
        X_data, y_data = X_data.to(device=device), y_data.to(device=device)
        X_train, y_train, X_val, y_val = split(X_data, y_data, ratio=0.8)

        for _ in t:
            train_loss = 0.0
            self.net.train()
            for X, y in zip(X_train, y_train):
                X = X.long()
                y_hat = self.net(X)
                y_hat = y_hat.squeeze(0)
                y = y.long()
                loss = F.cross_entropy(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(X_train)

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in zip(X_val, y_val):
                    X = X.long()
                    y_hat = self.net(X)
                    y_hat = y_hat.squeeze(0)
                    y = y.long()
                    loss = F.cross_entropy(y_hat, y)
                    val_loss += loss.item()
                val_loss /= len(X_val)

            t.set_description(
                f"Train loss: {train_loss:.3f}; Validation_loss: {val_loss:.3f};"
            )
            torch.save(self.net.state_dict(), checkpt_path)
        print(f"Checkpt saved at {checkpt_path}")

from torch import nn


class NGram(nn.Module):
    """
    A simple neural network with 3 layers: Embedding, a bi-directional LSTM & an output fully-connected layer.
    """

    def __init__(self, vocab_size) -> None:
        super(NGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.lstm = nn.LSTM(vocab_size, vocab_size, bidirectional=True)
        self.fc = nn.Linear(vocab_size * 2, vocab_size)

    def forward(self, idx):
        emb = self.embedding(idx)
        lstm, _ = self.lstm(emb)
        return self.fc(lstm[-1:,])

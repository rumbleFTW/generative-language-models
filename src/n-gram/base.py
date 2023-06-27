import torch
import torch.nn as nn
import json

### --- Globals
checkpt_path = "../../checkpts/ngram.pt"
index_path = "../../index_tables/ngram.json"
### --- Hyperparameters
epochs = 10
n = 8
split_ratio = 0.8
output_tokens = 30
### ---

device = "cuda" if torch.cuda.is_available() else "cpu"


class Tokenizer:
    """
    A string level tokenizer. It assigns an index to each string and stores the relation in an inex table.
    """

    def __init__(self, sequence=None):
        self.alien_token = 13333337
        if sequence:
            self.vocab = list(set(sequence))
            self.vocab_size = len(self.vocab)
            self.index_table = {string: idx for idx, string in enumerate(self.vocab)}
            self.index_table_inv = {
                idx: string for idx, string in enumerate(self.vocab)
            }

    def save(self, path):
        with open(path, "w") as table:
            json.dump(self.index_table, table)

    def load(self, path):
        with open(path, "r") as table:
            self.index_table = json.load(table)
            self.index_table_inv = {
                value: key for key, value in self.index_table.items()
            }
            self.vocab_size = len(self.index_table)

    def encode(self, sequence):
        return [self.index_table[string] for string in sequence]

    def decode(self, sequence):
        return [self.index_table_inv[token] for token in sequence]


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

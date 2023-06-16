import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import argparse
import re

from base import *


def data_loader(sequences, block_size, padding=0):
    X_data = []
    y_data = []
    for sentence in sequences:
        for i in range(1, len(sentence)):
            if i <= block_size:
                X = [padding] * (block_size - i) + sentence[:i]
            else:
                X = sentence[i - block_size : i]
            X_data.append(X)
            y_data.append(sentence[i])
    return (
        torch.tensor(X_data, dtype=torch.float32, requires_grad=True),
        torch.tensor(y_data),
    )


def split(X_data, y_data, ratio):
    split_point = int(len(X_data) * ratio)

    X_train = X_data[:split_point]
    y_train = y_data[:split_point]

    X_val = X_data[split_point:]
    y_val = y_data[split_point:]

    return X_train, y_train, X_val, y_val


def train(data, epochs):
    print(f"Training on {data} for {epochs} epochs")
    global tokenizer, ngram
    with open(data, "r") as f:
        raw_data = f.read()

    raw_data = re.sub(r"[^a-zA-Z ]", " ", raw_data.replace("\n", "").lower())
    raw_data_seq = [string.split(" ") for string in raw_data.split(".")]
    tokenizer = Tokenizer(raw_data.split(" "))
    tokenizer.save(index_path)
    sequence = tokenizer.encode(raw_data_seq[0])

    X_data, y_data = data_loader(sequences=[sequence], block_size=block_size)

    X_data, y_data = X_data.to(device=device), y_data.to(device=device)
    X_train, y_train, X_val, y_val = split(X_data, y_data, ratio=split_ratio)

    ngram = NGram(vocab_size=tokenizer.vocab_size)
    ngram = ngram.to(device=device)
    optimizer = torch.optim.Adam(ngram.parameters())

    t = tqdm(range(epochs))

    for epoch in t:
        for X, y in zip(X_train, y_train):
            X = X.long()
            y_hat = ngram(X)
            y_hat = y_hat.squeeze(0)
            y = y.long()
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t.set_description(f"Loss: {loss}")

    torch.save(ngram.state_dict(), checkpt_path)
    print(f"Checkpt saved at {checkpt_path}")


def generate(seed_text, output_tokens):
    seed_sequence = tokenizer.encode(seed_text.lower().split(" "))
    res = seed_sequence[:]
    seed_sequence = [0] * (block_size - len(seed_sequence)) + seed_sequence
    seed_tensor = torch.tensor(
        seed_sequence,
        dtype=torch.long,
        device=device,
    )
    for token in range(output_tokens):
        pred = torch.argmax(ngram(seed_tensor)).item()
        seed_sequence.append(pred)
        res.append(pred)
        seed_sequence = seed_sequence[-block_size:]
        seed_tensor = torch.tensor(
            seed_sequence,
            dtype=torch.long,
            device=device,
        )
    print(" ".join(tokenizer.decode(res)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data .txt file")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")

    args = parser.parse_args()

    train(data=args.data, epochs=args.epochs)

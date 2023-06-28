import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import re


from base import *


def generate_ngrams(sequences, n=2):
    """
    Split sequence of tokens `sequences` into n-grams of size `n`.
    """
    X_data = []
    y_data = []
    for i in range(len(sequences) - n):
        X_data.append(sequences[i : i + n])
        y_data.append(sequences[i + n])

    return (
        torch.tensor(X_data, dtype=torch.float32, requires_grad=True),
        torch.tensor(y_data),
    )


def preprocess_text(raw_text):
    """
    Cleans `raw_text`. This can be modified to include/exclude punctuations, special characters and uppercase letters.
    """
    raw_text = re.sub(r"[^a-zA-Z ]", " ", raw_text.replace("\n", " ").lower())
    return re.sub(r"\s{2,}", " ", raw_text)


def split(X_data, y_data, ratio):
    """
    Split data into training and testing set with ratio `ratio`.
    """
    split_point = int(len(X_data) * ratio)

    X_train = X_data[:split_point]
    y_train = y_data[:split_point]

    X_val = X_data[split_point:]
    y_val = y_data[split_point:]

    return X_train, y_train, X_val, y_val


def train(data, epochs, char_level=False):
    """
    Train the network on `data` for `epochs` number of epochs.
    """
    print(f"Training on {data} for {epochs} epochs")
    global tokenizer, ngram
    with open(data, "r") as f:
        raw_data = f.read()

    if char_level:
        tokenizer = Tokenizer(list(raw_data))
        sequence = tokenizer.encode(list(raw_data))
    else:
        raw_data = preprocess_text(raw_text=raw_data)
        tokenizer = Tokenizer(raw_data.split(" "))
        sequence = tokenizer.encode(raw_data.split(" "))
    tokenizer.save(index_path)
    X_data, y_data = generate_ngrams(sequences=sequence, n=n)

    X_data, y_data = X_data.to(device=device), y_data.to(device=device)
    X_train, y_train, X_val, y_val = split(X_data, y_data, ratio=split_ratio)

    ngram = NGram(vocab_size=tokenizer.vocab_size)
    ngram = ngram.to(device=device)
    optimizer = torch.optim.Adam(ngram.parameters())

    t = tqdm(range(epochs))

    for _ in t:
        train_loss = 0.0
        ngram.train()
        for X, y in zip(X_train, y_train):
            X = X.long()
            y_hat = ngram(X)
            y_hat = y_hat.squeeze(0)
            y = y.long()
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(X_train)

        ngram.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in zip(X_val, y_val):
                X = X.long()
                y_hat = ngram(X)
                y_hat = y_hat.squeeze(0)
                y = y.long()
                loss = F.cross_entropy(y_hat, y)
                val_loss += loss.item()
            val_loss /= len(X_val)

        t.set_description(
            f"Train loss: {train_loss:.3f}; Validation_loss: {val_loss:.3f};"
        )
        torch.save(ngram.state_dict(), checkpt_path)
    print(f"Checkpt saved at {checkpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data .txt file")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--char", action="store_true", help="character level training")

    args = parser.parse_args()

    train(data=args.data, epochs=args.epochs, char_level=args.char)

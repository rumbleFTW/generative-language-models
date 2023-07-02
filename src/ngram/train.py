import torch
import argparse

from tokenizer import Tokenizer
from ngram import NGram

import sys

sys.path.append("../")
from utils.preprocess import clean_text


### --- Globals
N = 5
INDEX_PATH = "../../index_tables/ngram.json"
CHECKPT_PATH = "../../checkpts/ngram.pt"


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


def train(data, epochs, level="word"):
    """
    Train the network on `data` for `epochs` number of epochs.
    """
    print(f"Training on {data} for {epochs} epochs")
    global tokenizer, ngram
    with open(data, "r") as f:
        raw_data = f.read()

    tokenizer = Tokenizer()
    if level == "char":
        tokenizer.fit(list(raw_data))
        sequence = tokenizer.encode(raw_data)
    elif level == "word":
        raw_data = clean_text(raw_data)
        tokenizer.fit(raw_data.split(" "))
        sequence = tokenizer.encode(raw_data)
    tokenizer.save(INDEX_PATH)
    X_data, y_data = generate_ngrams(sequences=sequence, n=N)

    ngram = NGram(vocab_size=tokenizer.vocab_size)
    ngram.fit(X_data, y_data, checkpt_path=CHECKPT_PATH, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data .txt file")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--l", type=str, help="level of generation")

    args = parser.parse_args()

    train(data=args.data, epochs=args.epochs, level=args.l)

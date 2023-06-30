import torch
import argparse
import time

from tokenizer import Tokenizer
from ngram import NGram


### --- Globals
N = 5
INDEX_PATH = "../../index_tables/ngram.json"
CHECKPT_PATH = "../../checkpts/ngram.pt"


def generate(seed_text, output_tokens, tokenizer):
    """
    Generate `output_tokens` numbers of tokens from seed `seed_text`.
    """
    seed_sequence = tokenizer.encode(seed_text)
    res = seed_sequence[:]
    seed_sequence = [0] * (N - len(seed_sequence)) + seed_sequence
    seed_tensor = torch.tensor(
        seed_sequence,
        dtype=torch.long,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for _ in range(output_tokens):
        pred = torch.argmax(ngram.net(seed_tensor)).item()
        seed_sequence.append(pred)
        res.append(pred)
        seed_sequence = seed_sequence[-N:]
        seed_tensor = torch.tensor(
            seed_sequence,
            dtype=torch.long,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if tokenizer.type == "char_level":
            print("".join(tokenizer.decode(res)), end="\r")
            time.sleep(0.05)
        elif tokenizer.type == "word_level":
            print(" ".join(tokenizer.decode(res)), end="\r")
            time.sleep(0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, help="seed string")
    parser.add_argument("--tokens", type=int, help="number of tokens to generate")
    parser.add_argument("--l", type=str, help="level of generation")

    args = parser.parse_args()

    if args.l == "char":
        tokenizer = Tokenizer(type="char_level")
        tokenizer.load(INDEX_PATH)
    elif args.l == "word":
        tokenizer = Tokenizer(type="word_level")
        tokenizer.load(INDEX_PATH)
    ngram = NGram(vocab_size=tokenizer.vocab_size)
    ngram.net.to(device="cuda" if torch.cuda.is_available() else "cpu")
    ngram.net.load_state_dict(
        torch.load(
            CHECKPT_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    generate(seed_text=args.seed, output_tokens=args.tokens, tokenizer=tokenizer)

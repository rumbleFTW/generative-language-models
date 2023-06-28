import torch
import argparse
import time

from base import *


def generate(seed_text, output_tokens, char_level=False):
    """
    Generate `output_tokens` numbers of tokens from seed `seed_text`.
    """
    if char_level:
        seed_sequence = tokenizer.encode(list(seed_text))
    else:
        seed_sequence = tokenizer.encode(seed_text.lower().split(" "))
    res = seed_sequence[:]
    seed_sequence = [0] * (n - len(seed_sequence)) + seed_sequence
    seed_tensor = torch.tensor(
        seed_sequence,
        dtype=torch.long,
        device=device,
    )
    for _ in range(output_tokens):
        pred = torch.argmax(ngram(seed_tensor)).item()
        seed_sequence.append(pred)
        res.append(pred)
        seed_sequence = seed_sequence[-n:]
        seed_tensor = torch.tensor(
            seed_sequence,
            dtype=torch.long,
            device=device,
        )
        if char_level:
            print("".join(tokenizer.decode(res)), end="\r")
            time.sleep(0.05)
        else:
            print(" ".join(tokenizer.decode(res)), end="\r")
            time.sleep(0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, help="seed string")
    parser.add_argument("--tokens", type=int, help="number of tokens to generate")
    parser.add_argument(
        "--char", action="store_true", help="character level generation"
    )

    args = parser.parse_args()

    tokenizer = Tokenizer()
    tokenizer.load(index_path)

    ngram = NGram(vocab_size=tokenizer.vocab_size)
    ngram.to(device=device)
    ngram.load_state_dict(torch.load(checkpt_path, map_location=device))
    generate(seed_text=args.seed, output_tokens=args.tokens, char_level=args.char)

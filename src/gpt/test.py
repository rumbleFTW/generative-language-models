from skipgram import SkipGram
import sys

sys.path.append("../")

from utils.preprocess import clean_text

with open("../../data/gita_chap1.txt", "r") as f:
    raw_data = clean_text(f.read().replace("\n", " "))

s = SkipGram(embedding_dim=2, context_size=3)

s.fit(raw_data.split(" "), epochs=100)

s.save("../../index_tables/gpt.json")

from skipgram import SkipGram
import sys

sys.path.append("../")

from utils.preprocess import clean_text

with open("../../data/test.txt", "r") as f:
    raw_data = clean_text(f.read().replace("\n", " "))

print(raw_data)
s = SkipGram(2, 5)

s.fit(raw_data.split(" "), epochs=10)

s.save("../../index_tables/gpt.json")

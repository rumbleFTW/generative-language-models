import json


class Tokenizer:
    """
    A string level tokenizer. It assigns an index to each string and stores the relation in an inex table.
    """

    def __init__(self, corpus: str = None, type: str = "word_level"):
        if corpus:
            if type == "word_level":
                self.vocab = list(set(corpus.split(" ")))
            elif type == "char_level":
                self.vocab = list(set(list(corpus)))
            self.vocab_size = len(self.vocab)
            self.index_table = {string: idx for idx, string in enumerate(self.vocab)}
            self.index_table_inv = {
                idx: string for idx, string in enumerate(self.vocab)
            }
        self.type = type

    def save(self, path: str):
        with open(path, "w") as table:
            json.dump(self.index_table, table)

    def load(self, path: str):
        with open(path, "r") as table:
            self.index_table = json.load(table)
            self.index_table_inv = {
                value: key for key, value in self.index_table.items()
            }
            self.vocab_size = len(self.index_table)

    def encode(self, sequence: iter):
        if self.type == "word_level":
            return [self.index_table[string] for string in sequence.split(" ")]
        elif self.type == "char_level":
            return [self.index_table[string] for string in list(sequence)]

    def decode(self, sequence: iter):
        return [self.index_table_inv[token] for token in sequence]

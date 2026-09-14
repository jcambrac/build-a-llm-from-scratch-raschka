import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
        print("int_to_str", self.int_to_str)

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[idx] for idx in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
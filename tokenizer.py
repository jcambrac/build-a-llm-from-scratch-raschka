import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed] # unknown token    
        ids = [self.str_to_int[item] for item in preprocessed]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[idx] for idx in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
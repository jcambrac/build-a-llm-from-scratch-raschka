import urllib.request
import re
from tokenizer import SimpleTokenizer

url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")

file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r") as file:
    raw_text = file.read()

print("Total characters in the file:", len(raw_text))
print("First 500 characters:", raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

print(len(preprocessed))
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
# all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)

vocab = {token: idx for idx, token in enumerate(all_words)}
print("vocab")

for i, item in enumerate(vocab.items()):
    print( item)
    if i >= 50:
        break

tokenizer = SimpleTokenizer(vocab)

text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

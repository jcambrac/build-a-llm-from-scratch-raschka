import urllib.request
import re
from tokenizer import SimpleTokenizer
import tiktoken

url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")

file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r") as file:
    raw_text = file.read()

print("Total characters in the file:", len(raw_text))
# print("First 500 characters:", raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

print(len(preprocessed))
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))
# print(preprocessed[:30])

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
# print("Vocabulary size:", vocab_size)

vocab = {token: idx for idx, token in enumerate(all_words)}
# print("vocab")

print(list(vocab.items())[-5:])

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

tokenizer = SimpleTokenizer(vocab)

text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizer(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

unknown_strange_word = "akwirdamn"
integers = tokenizer.encode(unknown_strange_word)
print(integers)

strings = tokenizer.decode(integers)
print(strings)

for item in integers:
    print( tokenizer.decode([item]) , "->" , item)

# print(raw_text)

encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))

encoded_sample = encoded_text[50:]
# print(encoded_sample)

context_size = 4
x = encoded_sample[:context_size]
y = encoded_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range (1 , context_size + 1):
    context = encoded_sample[:i]
    target = encoded_sample[i]
    print(f"context: {context} -> {target}")

print("--------------------------------")

for i in range(1 , context_size + 1):
    string_context = tokenizer.decode(encoded_sample[:i])
    string_target = tokenizer.decode([encoded_sample[i]])
    print(f"context: {string_context} -> {string_target}")
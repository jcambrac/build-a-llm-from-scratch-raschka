from dataset import GPTDataset
import tiktoken
from dataset import create_dataloader
tokenizer = tiktoken.get_encoding("gpt2")

dataset = GPTDataset(
    txt = "Hello, how are you? I am fine, thank you.",
    tokenizer = tokenizer,
    max_length = 4,
    stride = 2
)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

dataiter = iter(dataloader)
first_batch = next(dataiter)
print(first_batch)
second_batch = next(dataiter)
print(second_batch)

dataloader = create_dataloader(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

print("printing 8 batch:")
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
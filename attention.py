import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] #  get the second input vector as the query

print("query:", query)

attn_scores_2 = torch.empty(inputs.shape[0]) # (6,) initialize empty tensor to store attention scores

print("attn_scores_2:", attn_scores_2)

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("attn_scores_2:", attn_scores_2)

res = 0
print("inputs[0]:", inputs[0])
print ("--------------------------------")

for idx, element in enumerate(inputs[0]):
    print("idx:", idx , "element:", element , "query[idx]:", query[idx])
    res += inputs[0][idx] * query[idx]
print("res:", res)
print("torch.dot(inputs[0], query):", torch.dot(inputs[0], query))
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

print ("--------------------------------")
print("attn_scores_2:", attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum of Attention weights:", attn_weights_2_tmp.sum())

def softmax_naive(x):
  return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_softmax = softmax_naive(attn_scores_2)
print("Attention weights softmax:", attn_weights_2_softmax)
print("Sum of Attention weights softmax:", attn_weights_2_softmax.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0) # finally attention weights use the standard softmax function
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

print ("--------------------------------")

print("inputs:", inputs)
query = inputs[1]         #1
context_vec_2 = torch.zeros(query.shape)
print("context_vec_2:", context_vec_2)
for i, x_i in enumerate(inputs): # loop through each input vector
  print("i:", i , "x_i:", x_i)
  context_vec_2 += attn_weights_2[i]*x_i # multiply the attention weight by the input vector and add to the context vector
print("context_vec_2:", context_vec_2)

# the context vector is the weighted sum of the input vectors, where the weights are the attention weights
# attention weights are calculated by the dot product of the query and the input vectors
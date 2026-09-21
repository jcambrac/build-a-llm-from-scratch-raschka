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

attn_scores = torch.empty(inputs.shape[0], inputs.shape[0]) # (6, 6) initialize empty tensor to store attention scores between all input vectors
for i, x_i in enumerate(inputs): # loop through each input vector
  for j, x_j in enumerate(inputs): # loop through each input vector
    attn_scores[i, j] = torch.dot(x_i, x_j) # calculate the dot product of the input vectors
print(attn_scores)

attn_scores = inputs @ inputs.T # (6, 6) calculate the attention scores between all input vectors using matrix multiplication
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1) # normalize the attention scores to get the attention weights # dim=-1 means the last dimension of the tensor, in this case the rows
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]) # sum of the second row of the attention weights
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1)) # sum of all the rows of the attention weights # this should be 1 for each row

all_context_vecs = attn_weights @ inputs # (6, 3) calculate the context vector for each input vector using the attention weights and the input vectors
print(all_context_vecs)

print("Previous 2nd context vector:", context_vec_2 , "is equal to the 2nd context vector in all_context_vecs:", all_context_vecs[1] ,  all_context_vecs[1] == context_vec_2)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

print("x_2:", x_2)
print("d_in:", d_in)

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query:", W_query)

query_2 = x_2 @ W_query 
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
print("query_2:", query_2)

keys = inputs @ W_key 
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("keys:", keys)
print("values.shape:", values.shape)
import torch as torch
import random
import numpy
#GAP = '*'
#UNKNOWN = 'N'
#ALPHABET = 'ACGT' + GAP + UNKNOWN
#encoding = {v: i for i, v in enumerate(ALPHABET)}
#decoding = {v: k for k, v in encoding.items()}
#print("encoding",encoding) #encoding {'A': 0, 'C': 1, 'G': 2, 'T': 3, '*': 4, 'N': 5}
#reverse = forward + 6

torch.set_printoptions(profile="full")
X = torch.randint(0,12,(2,3,4))

# generate a boolean mask
probs=torch.rand(X.size())
token = probs > 0.85
print("token '\n'", token)

rand = probs > 0.9
print("rand '\n'", rand)

# get indices of the masked bases
token_indices = (token == True).nonzero(as_tuple=False).numpy()
rand_indices = (rand == True).nonzero(as_tuple=False).numpy()

print("before masking x '\n'",X)
X_before_masking = X[token].clone()
print("before applying mask token '\n'", X_before_masking)

for h in token_indices:
    print("before token masking ", X[h[0]][h[1]][h[2]])
    X[h[0]][h[1]][h[2]] = torch.tensor([5]) if X[h[0]][h[1]][h[2]] < 6 else torch.tensor([11])
    print("token masking at index: ", h)
    print("after token masking", X[h[0]][h[1]][h[2]])

print("after token masking '\n'",X)

for j in rand_indices:
    print("before rand masking ", X[j[0]][j[1]][j[2]])
    X[j[0]][j[1]][j[2]] = torch.tensor([random.choice(range(0,6))]) if X[j[0]][j[1]][j[2]] < 6 else torch.tensor([random.choice(range(6,12))])
    print("rand masking at index: ", j)
    print("after randmasking", X[j[0]][j[1]][j[2]])

print("masked using random token'\n'",X)
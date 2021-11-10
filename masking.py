import torch as torch
import random
import numpy

torch.set_printoptions(profile="full")
X1 = torch.randint(0,6,(1,3,4))
X2 = torch.randint(6,12,(1,3,4))
X = torch.cat((X1,X2),dim=0)
print("input x \n",X)

# generate a boolean mask
probs=torch.rand(X.size())

token = probs > 0.85
print("token \n", token)

rand = probs > 0.9
print("rand \n", rand)

X_original = X[token].clone()

#X = X % 6
print("before masking x \n",X)

#print("before applying mask token \n", X_before_masking)

X[token] = torch.full(X[token].size(),5) + X[token]//6*6
print("after token masking \n", X)

X[rand] = torch.randint(0, high = 6, size=X[rand].size()) + X[rand]//6*6
print("after rand masking \n", X)
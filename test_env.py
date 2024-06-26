import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)

x = torch.Tensor([2.1])
xx = x.cuda()
print(xx)

# CUDNN TEST
from torch.backends import cudnn

print('cudnn is ' + str(cudnn.is_acceptable(xx)))
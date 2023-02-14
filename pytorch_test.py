import torch

x = torch.rand(5, 3)
print(x)

import torch.nn as nn

m = nn.LeakyReLU(0.1)
input = torch.randn(2)
output = m(input)
print(output)

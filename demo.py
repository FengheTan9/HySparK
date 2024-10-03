import torch
noise = torch.Tensor([1, 0, 0, 1, 0, 1, 1, 0, 0])
indices = torch.arange(noise.size(1), device='cuda').reshape(1, -1, 1)
print(torch.argsort(noise, dim=0))
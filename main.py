import torch
from nogeometric import HGNN

x = torch.ones(17)
x = torch.diag(x)

hyperedge_index = {2:torch.tensor([[0, 1, 3, 4, 6, 7],
                                [2, 2, 5, 5, 8, 8]], dtype=torch.long),
                   3:torch.tensor([[9, 10, 11, 13, 14, 15],
                                [12, 12, 12, 16, 16, 16]], dtype=torch.long)}

model = HGNN()
out = model(x, hyperedge_index)
print(out)
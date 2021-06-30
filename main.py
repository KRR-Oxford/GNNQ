import torch
from nogeometric import HGNN


base_dim = 17
x = torch.ones(base_dim)
x = torch.diag(x)
# x = torch.rand((20,5))

hyperedge_index = { 1:torch.tensor([[0, 1, 2, 3, 4, 5],
                                [6, 7, 8, 9, 10, 11]], dtype=torch.long),
                    2:torch.tensor([[0, 1, 3, 4, 6, 7],
                                [2, 2, 5, 5, 8, 8]], dtype=torch.long),
                    3:torch.tensor([[9, 10, 11, 13, 14, 15],
                                [12, 12, 12, 16, 16, 16]], dtype=torch.long)}

hyperedge_type = {1:torch.tensor([1,1,1,1,2,2]),2:torch.tensor([0,1,2]), 3:torch.tensor([0,1])}

num_edge_types_by_shape = {1:3, 2:3, 3:3}

model = HGNN(base_dim, num_edge_types_by_shape)
out = model(x, hyperedge_index, hyperedge_type)
print(out)
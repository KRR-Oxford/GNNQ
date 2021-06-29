import torch
import torch.nn as nn
import torch_scatter as scatter


class HGNN(nn.Module):
    def __init__(self, base_dim, num_edge_types_by_shape):
        super(HGNN, self).__init__()
        self.base_dim = base_dim
        self.num_edge_types_by_shape = num_edge_types_by_shape
        self.weights = {}
        # Shape is the number of src nodes -- we consider edges of the form ((u_0,...,u_{n-1}), R, v)
        for shape in num_edge_types_by_shape:
            self.weights[shape] = torch.nn.Parameter(torch.zeros(num_edge_types_by_shape[shape], base_dim * shape, base_dim))
            # Dummy weights for debugging
            weight = torch.diag(torch.ones(base_dim)).repeat(shape, 1).repeat(num_edge_types_by_shape[shape],1,1)
            for i in range(1, num_edge_types_by_shape[shape] + 1):
                weight[i-1] = weight[i-1] * i
            self.weights[shape] = weight

    def forward(self,  x, hyperedge_index, hyperedge_type):
        index = torch.tensor([],dtype=torch.int16)
        x_j = torch.tensor([])
        # Loop through hyperedges with different shapes (num of src nodes)
        for shape in hyperedge_index:
            # Loop through edge_types for every shape
            for edge_type in range(0, self.num_edge_types_by_shape[shape]):
                mask = (hyperedge_type[shape] == edge_type)
                mask = mask.unsqueeze(1).repeat(1,shape).flatten()
                ind = torch.reshape(hyperedge_index[shape][1][mask], (-1, shape))
                if (ind.nelement() != 0):
                    ind = ind[:, 0]
                    index = torch.cat((index, ind), dim=0)
                    tmp = x[hyperedge_index[shape][0][mask]]
                    s = tmp.size()[1] * shape
                    # Concat feature vectors of src nodes for hyperedges
                    concat = torch.reshape(tmp, (-1, s))
                    weight = self.weights[shape][edge_type]
                    transformed = torch.mm(concat, weight)
                    # Tensor with all messages
                    x_j = torch.cat((x_j, transformed))
        index = index.unsqueeze(1).expand_as(x_j)
        # Aggregate function
        out = scatter.scatter_add(src=x_j, index=index, out=torch.zeros_like(x), dim=0)
        # Combine function
        out = x + out
        return out


import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_scatter as scatter


class HGNNLayer(nn.Module):
    def __init__(self, base_dim, num_edge_types_by_shape):
        super(HGNNLayer, self).__init__()
        self.base_dim = base_dim
        self.num_edge_types_by_shape = num_edge_types_by_shape
        self.lin = {}
        # Shape is the number of src nodes -- we consider edges of the form ((u_0,...,u_{n-1}), R, v)
        for shape in num_edge_types_by_shape:
            self.lin[shape] = torch.nn.Parameter(
                torch.zeros(num_edge_types_by_shape[shape], base_dim * shape, base_dim))
            nn.init.xavier_normal(self.lin[shape])
            # Dummy weights for debugging
            weight = torch.diag(torch.ones(base_dim)).repeat(shape, 1).repeat(num_edge_types_by_shape[shape], 1, 1)
            for i in range(1, num_edge_types_by_shape[shape] + 1):
                weight[i - 1] = weight[i - 1] * i
            self.lin[shape] = weight

    def forward(self, x, hyperedge_index, hyperedge_type):
        index = torch.tensor([], dtype=torch.int16)
        x_j = torch.tensor([])
        # Loop through hyperedges with different shapes (num of src nodes)
        for shape in hyperedge_index:
            # Loop through edge_types for every shape
            for edge_type in range(0, self.num_edge_types_by_shape[shape]):
                mask = (hyperedge_type[shape] == edge_type)
                mask = mask.unsqueeze(1).repeat(1, shape).flatten()
                i = torch.reshape(hyperedge_index[shape][1][mask], (-1, shape))
                if (i.nelement() != 0):
                    # Compute indices for scatter function
                    i = i[:, 0]
                    index = torch.cat((index, i), dim=0)
                    # Compute src for scatter function
                    tmp = x[hyperedge_index[shape][0][mask]]
                    s = tmp.size()[1] * shape
                    # Concat feature vectors of src nodes for hyperedges
                    concatted = torch.reshape(tmp, (-1, s))
                    weights = self.lin[shape][edge_type]
                    transformed = torch.mm(concatted, weights)
                    # Normalisation
                    _, revers_i, counts = torch.unique(i, return_inverse=True, return_counts=True)
                    norm = 1 / counts[revers_i]
                    transformed = norm.unsqueeze(1) * transformed
                    # Tensor with all messages
                    x_j = torch.cat((x_j, transformed))
        index = index.unsqueeze(1).expand_as(x_j)
        # Aggregate function
        out = scatter.scatter_add(src=x_j, index=index, out=torch.zeros_like(x), dim=0)
        # Combine function
        out = x + out
        return out


class HGNN(nn.Module):
    def __init__(self, base_dim, num_edge_types_by_shape):
        super(HGNN, self).__init__()
        self.num_edge_types_by_shape = num_edge_types_by_shape
        self.layer1 = HGNNLayer(base_dim, num_edge_types_by_shape)
        self.activation1 = f.relu
        self.layer2 = HGNNLayer(base_dim, num_edge_types_by_shape)
        self.activation2 = f.relu

    def forward(self, x, hyperedge_index, hyperedge_type):
        h = self.layer1(x, hyperedge_index, hyperedge_type)
        h = self.activation1(h)
        h = self.layer2(h, hyperedge_index, hyperedge_type)
        h = self.activation2(h)
        return h

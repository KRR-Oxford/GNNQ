import torch
import torch.nn as nn
import torch_scatter as scatter
from collections import defaultdict


# Definition of a HGNN-layer
class HGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, shapes_dict):
        super().__init__()
        self.shapes_dict = shapes_dict
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Includes bias
        self.C = torch.nn.Linear(input_dim, output_dim)
        self.A = torch.nn.ParameterDict({})
        # Shape is the number of nodes in a hyperedge minus 1 -- we consider edges of the form (v, R, (u_0,...,u_{n-1}))
        shape_nums = defaultdict(int)
        for _, shape in shapes_dict.items():
            shape_nums[shape] += 1
        for edge, shape in shapes_dict.items():
            self.A[edge] = torch.nn.Parameter(torch.zeros(input_dim * shape, output_dim))
            nn.init.normal_(self.A[edge], mean=0, std=torch.sqrt(
                torch.tensor(2 / (shape_nums[shape] * input_dim * shape + shape_nums[shape] * output_dim))))

    # Computes feature vectors for one message passing step
    def forward(self, x, indices_dict, device):
        dest_indices = torch.tensor([], dtype=torch.int16, device=device)
        msgs = torch.tensor([], dtype=torch.float16, device=device)
        x = x.to(device)
        # Loops through all edge types
        for edge, edge_indices in indices_dict.items():
            # Check whether edge type is in shapes dict and the graph contains edges of this type
            edge_indices = edge_indices.to(device)
            if (edge in self.shapes_dict.keys()) and edge_indices.numel():
                i = torch.reshape(edge_indices[1], (-1, self.shapes_dict[edge]))
                # Computes indices for scatter function
                i = i[:, 0]
                dest_indices = torch.cat((dest_indices, i), dim=0)
                # Computes src for scatter function
                tmp = x[edge_indices[0]]
                s = tmp.size()[1] * self.shapes_dict[edge]
                # Concats feature vectors of src nodes for hyperedges
                tmp = torch.reshape(tmp, (-1, s))
                weights = self.A[edge]
                # Linear transformation of messages
                tmp = tmp.mm(weights)
                # Normalises elements of the messages (vectors)
                _, revers_i, counts = torch.unique(i, return_inverse=True, return_counts=True)
                norm = 1 / counts[revers_i]
                tmp = norm.unsqueeze(1) * tmp
                # Constructs tensor containing all messages
                msgs = torch.cat((msgs, tmp))
        # Expands tensor containing the destionation indices to the dimensions of the message tensor
        dest_indices = dest_indices.unsqueeze(1).expand_as(msgs)
        # Aggregates all messages
        agg = scatter.scatter_add(src=msgs, index=dest_indices, out=torch.zeros(x.size()[0], self.output_dim, device=device),
                                      dim=0)
        # Combines aggregated message vector with the vector from the previous layer
        h = self.C(x) + agg
        return h


# Definition of the HGNN-model
class HGNN(nn.Module):
    def __init__(self, query_string, feat_dim, base_dim, shapes_dict, num_layers, negative_slope=0.1, subqueries=None):
        super().__init__()
        self.shapes_dict = shapes_dict
        self.query_string = query_string
        self.subqueries = subqueries
        self.num_layers = num_layers
        self.negative_slope = negative_slope
        self.msg_layers = nn.ModuleList([])
        self.msg_layers.append(HGNNLayer(feat_dim, base_dim, shapes_dict))
        for i in range(1, self.num_layers - 1):
            self.msg_layers.append(HGNNLayer(base_dim, base_dim, shapes_dict))
        self.msg_layers.append(HGNNLayer(base_dim, 1, shapes_dict))

    def forward(self, x, indices_dict, device, logits=False):
        # Loops through all message passing layers
        for i in range(self.num_layers - 1):
            x = self.msg_layers[i](x, indices_dict, device)
            x = nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.msg_layers[self.num_layers - 1](x, indices_dict, device)
        if logits: return x
        return torch.sigmoid(x)

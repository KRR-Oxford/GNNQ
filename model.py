import torch
import torch.nn as nn
import torch_scatter as scatter
from collections import defaultdict


# Implementation of a HGNN layer
class HGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, shapes_dict, max_aggr):
        super().__init__()
        self.shapes_dict = shapes_dict
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_aggr = max_aggr
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

    # This function computes the updated feature vectors for one message passing iteration
    def forward(self, x, indices_dict):
        dest_indices = torch.tensor([], dtype=torch.int16)
        msgs = torch.tensor([], dtype=torch.float16)
        # Loop through all edge types
        for edge, edge_indices in indices_dict.items():
            # Check whether edge type is in shapes dict and the graph contains edges of this type
            if (edge in self.shapes_dict.keys()) and edge_indices.numel():
                i = torch.reshape(edge_indices[1], (-1, self.shapes_dict[edge]))
                # Compute indices for scatter function
                i = i[:, 0]
                dest_indices = torch.cat((dest_indices, i), dim=0)
                # Compute src for scatter function
                tmp = x[edge_indices[0]]
                s = tmp.size()[1] * self.shapes_dict[edge]
                # Concat feature vectors of src nodes for hyperedges
                tmp = torch.reshape(tmp, (-1, s))
                weights = self.A[edge]
                # Linear transformation of messages
                tmp = tmp.mm(weights)
                # Normalise messages
                if not self.max_aggr:
                    _, revers_i, counts = torch.unique(i, return_inverse=True, return_counts=True)
                    norm = 1 / counts[revers_i]
                    tmp = norm.unsqueeze(1) * tmp
                # Construct tensor containing all messages
                msgs = torch.cat((msgs, tmp))
        # Expand tensor with dest indices to dimensions of the message tensor
        dest_indices = dest_indices.unsqueeze(1).expand_as(msgs)
        # Aggregate all messages
        if self.max_aggr:
            agg, _ = scatter.scatter_max(src=msgs, index=dest_indices, out=torch.zeros(x.size()[0], self.output_dim),
                                         dim=0)
        else:
            agg = scatter.scatter_add(src=msgs, index=dest_indices, out=torch.zeros(x.size()[0], self.output_dim),
                                      dim=0)
        # Combine message vector with the vector from the previous layer
        h = self.C(x) + agg
        return h


# Implementation of the HGNN model
class HGNN(nn.Module):
    def __init__(self, query_string, feat_dim, base_dim, shapes_dict, num_layers, negative_slope=0.1, max_aggr=False,
                 monotonic=False, subqueries=None):
        super().__init__()
        self.shapes_dict = shapes_dict
        self.query_string = query_string
        self.subqueries = subqueries
        self.num_layers = num_layers
        self.negative_slope = negative_slope
        self.monotonic = monotonic
        self.msg_layers = nn.ModuleList([])
        self.msg_layers.append(HGNNLayer(feat_dim, base_dim, shapes_dict, max_aggr))
        for i in range(1, self.num_layers - 1):
            self.msg_layers.append(HGNNLayer(base_dim, base_dim, shapes_dict, max_aggr))
        self.msg_layers.append(HGNNLayer(base_dim, 1, shapes_dict, max_aggr))

    # Function that sets all negative weights to zero
    def clamp_negative_weights(self):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                param.data.clamp_(0)

    def forward(self, x, indices_dict, logits=False):
        # Sets all negative weights to zero and thus forces network to be monotonic
        if self.monotonic:
            self.clamp_negative_weights()
        # Message passing layers
        for i in range(self.num_layers - 1):
            x = self.msg_layers[i](x, indices_dict)
            x = nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.msg_layers[self.num_layers - 1](x, indices_dict)
        if logits: return x
        return torch.sigmoid(x)

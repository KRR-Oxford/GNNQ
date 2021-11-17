import torch
import torch.nn as nn
import torch_scatter as scatter
from collections import defaultdict


class HGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, shapes_dict, max_aggr):
        super(HGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_aggr = max_aggr
        # Includes bias
        self.C = torch.nn.Linear(input_dim, output_dim)
        self.A = torch.nn.ParameterDict({})
        # Shape is the number of src nodes -- we consider edges of the form ((u_0,...,u_{n-1}), R, v)
        shape_nums = defaultdict(int)
        for _, shape in shapes_dict.items():
            shape_nums[shape]+=1
        for edge, shape in shapes_dict.items():
            self.A[edge] = torch.nn.Parameter(torch.zeros(input_dim * shape, output_dim))
            nn.init.normal_(self.A[edge], mean=0, std=torch.sqrt(torch.tensor(2/(shape_nums[shape] * input_dim * shape + shape_nums[shape] * output_dim))))

    # h = x_iC + b + SUM_shape SUM_type SUM_neighbours x_jA_shape,type
    def forward(self, x, indices_dict, shapes_dict):
        # Not sure whether these tensors are automatically move to device
        dest_indices = torch.tensor([], dtype=torch.int16)
        msgs = torch.tensor([], dtype=torch.float16)
        # Loop through all edge types (hyperedge types).
        for edge, edge_indices in indices_dict.items():
            i = torch.reshape(edge_indices[1], (-1, shapes_dict[edge]))
            if (i.nelement() != 0):
                # Compute indices for scatter function
                i = i[:, 0]
                dest_indices = torch.cat((dest_indices, i), dim=0)
                # Compute src for scatter function
                tmp = x[edge_indices[0]]
                s = tmp.size()[1] * shapes_dict[edge]
                # Concat feature vectors of src nodes for hyperedges
                tmp = torch.reshape(tmp, (-1, s))
                weights = self.A[edge]
                # Linear transformation of messages
                tmp = tmp.mm(weights)
                # Normalisation
                _, revers_i, counts = torch.unique(i, return_inverse=True, return_counts=True)
                norm = 1 / counts[revers_i]
                tmp = norm.unsqueeze(1) * tmp
                # Tensor with all messages
                msgs = torch.cat((msgs, tmp))
        dest_indices = dest_indices.unsqueeze(1).expand_as(msgs)
        # Aggregate
        if self.max_aggr:
            agg, _ = scatter.scatter_max(src=msgs, index=dest_indices, out=torch.zeros(x.size()[0], self.output_dim),
                                         dim=0)
        else:
            agg = scatter.scatter_add(src=msgs, index=dest_indices, out=torch.zeros(x.size()[0], self.output_dim),
                                      dim=0)
        # Combine
        h = self.C(x) + agg
        return h


class HGNN(nn.Module):
    def __init__(self, feat_dim, base_dim, shapes_dict, num_layers, negative_slope=0.01, max_aggr=False, monotonic=False):
        super(HGNN, self).__init__()
        self.num_layers = num_layers
        self.negative_slope = negative_slope
        self.monotonic= monotonic
        self.msg_layers = nn.ModuleList([])
        self.msg_layers.append(HGNNLayer(feat_dim, base_dim, shapes_dict, max_aggr))
        for i in range(1, self.num_layers - 1):
            self.msg_layers.append(HGNNLayer(base_dim, base_dim, shapes_dict, max_aggr))
        self.msg_layers.append(HGNNLayer(base_dim, 1, shapes_dict, max_aggr))

    def clamp_negative_weights(self):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                param.data.clamp_(0)

    def forward(self, x, indices_dict, shapes_dict, logits=False):
        if self.monotonic:
            self.clamp_negative_weights()
        # Message passing layers
        for i in range(self.num_layers - 1):
            x = self.msg_layers[i](x, indices_dict, shapes_dict)
            x = nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.msg_layers[self.num_layers - 1](x, indices_dict, shapes_dict)
        if logits: return x
        return torch.sigmoid(x)

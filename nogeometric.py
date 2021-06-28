import torch
import torch.nn as nn
import torch_scatter as scatter


class HGNN(nn.Module):
    def __init__(self):
        super(HGNN, self).__init__()

    def forward(self,  x, hyperedge_index):
        index = torch.tensor([],dtype=torch.int16)
        x_j = torch.tensor([])
        # Loop through hyperedges with i src nodes -- second loop for different relation types is required
        for i in hyperedge_index:
            ind = torch.reshape(hyperedge_index[i][1], (-1, i))
            ind = ind[:, 0]
            index = torch.cat((index, ind), dim=0)
            help = x[hyperedge_index[i][0]]
            s = help.size()[1] * i
            # Concat feature vectors of src nodes for hyperedges
            concat = torch.reshape(help, (-1, s))
            # Dummy weight vectors -- message function
            weight = torch.diag(torch.ones(x.size()[1])).repeat(i, 1)
            transformed = torch.mm(concat, weight)
            # Tensor with all messages
            x_j = torch.cat((x_j, transformed))
        index = index.unsqueeze(1).expand_as(x_j)
        # Aggregate function
        out = scatter.scatter_add(src=x_j, index=index, out=torch.zeros_like(x), dim=0)
        # Combine function
        out = x + out
        return out


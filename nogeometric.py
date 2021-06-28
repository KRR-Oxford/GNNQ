import torch
import torch.nn as nn
import torch_scatter as scatter


class HGNN(nn.Module):
    def __init__(self):
        super(HGNN, self).__init__()

    def forward(self,  x, hyperedge_index):
        print('Message')
        index = torch.tensor([],dtype=int)
        x_j = torch.tensor([])
        for i in hyperedge_index:
            ind = torch.reshape(hyperedge_index[i][1], (-1, i))
            ind = ind[:, 0]
            index = torch.cat((index, ind), dim=0)
            help = x[hyperedge_index[i][0]]
            s = help.size()[1] * i
            concat = torch.reshape(help, (-1, s))
            weight = torch.diag(torch.ones(x.size()[1])).repeat(i, 1)
            transformed = torch.mm(concat, weight)
            x_j = torch.cat((x_j, transformed))
        index = index.unsqueeze(1).expand_as(x_j)
        out = scatter.scatter_add(src=x_j, index=index, dim=0)
        out = x + out
        return out


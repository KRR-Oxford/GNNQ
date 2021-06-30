import torch
from nogeometric import HGNN


num_nodes = 17
base_dim = 17
epochs = 500
learning_rate = 0.1
x = torch.ones(base_dim)
x = torch.diag(x)
# x = torch.rand((num_nodes,base_dim))
y = torch.randint(0, 2, (num_nodes,),dtype=torch.float32)

hyperedge_index = { 1:torch.tensor([[0, 1, 2, 3, 4, 5],
                                [6, 6, 8, 9, 10, 11]], dtype=torch.long),
                    2:torch.tensor([[0, 1, 3, 4, 6, 7],
                                [2, 2, 5, 5, 8, 8]], dtype=torch.long),
                    3:torch.tensor([[9, 10, 11, 13, 14, 15],
                                [12, 12, 12, 16, 16, 16]], dtype=torch.long)}

hyperedge_type = {1:torch.tensor([1,1,1,1,2,2]),2:torch.tensor([0,1,2]), 3:torch.tensor([0,1])}

num_edge_types_by_shape = {1:5, 2:3, 3:3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HGNN(base_dim, num_edge_types_by_shape)
model.to(device)
x = x.to(device)
help = model.parameters()
for param in model.parameters():
    print(type(param.data), param.size())

# out = model(x, hyperedge_index, hyperedge_type)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

def train_loop(model, loss_fn, optimizer, epochs, hyperedge_index, hyperedge_type, x, y):
    for i in range(epochs):
        pred = model(x, hyperedge_index, hyperedge_type).flatten()
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_loop(model, loss_fn, optimizer, epochs, hyperedge_index, hyperedge_type,x,y)

out = model(x, hyperedge_index, hyperedge_type)
print(torch.cat((out, y.unsqueeze(1)), dim=1))
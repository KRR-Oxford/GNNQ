import torch
from model import HGNN
from data import data
import numpy as np

# hyperedge_index = { 1:torch.tensor([[0, 1, 2, 3, 4, 5],
#                                 [6, 6, 8, 9, 10, 11]], dtype=torch.long, device=device),
#                     2:torch.tensor([[0, 1, 3, 4, 6, 7],
#                                 [2, 2, 5, 5, 8, 8]], dtype=torch.long, device=device),
#                     3:torch.tensor([[9, 10, 11, 13, 14, 15],
#                                 [12, 12, 12, 16, 16, 16]], dtype=torch.long, device=device)}
#
# hyperedge_type = {1:torch.tensor([1,1,1,1,4,4]),2:torch.tensor([0,1,2]), 3:torch.tensor([0,1])}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

triples, entity2id, relation2id, id2entity, id2relation = data(['watdiv.ttl'])
with open('answers.npy', 'rb') as f:
    true_answers = torch.tensor(np.load(f))

num_nodes = len(entity2id)
base_dim = 40
num_layers = 4
epochs = 20
learning_rate = 0.1
x = torch.cat((torch.ones(num_nodes,1), torch.zeros(num_nodes,base_dim - 1)), dim=1)

# K - hot vector indicating all answers
y = torch.scatter(torch.zeros(num_nodes), 0, true_answers, torch.ones(num_nodes))

entities = torch.tensor(triples['watdiv.ttl']).t()[[0,2]]
entities = torch.cat((entities, entities[[1,0]]), dim=1)
relations = torch.tensor(triples['watdiv.ttl']).t()[1]
relations = torch.cat((relations, relations + torch.max(relations) + 1),dim=0)
hyperedge_index = {1: entities}
hyperedge_type = {1: relations}

num_edge_types_by_shape = {1:len(relation2id) * 2}

model = HGNN(base_dim, num_edge_types_by_shape, num_layers)
model.to(device)
help = model.parameters()
for param in model.parameters():
    print(type(param.data), param.size())

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

def train_loop(model, loss_fn, optimizer, epochs, hyperedge_index, hyperedge_type, x, y):
    for i in range(epochs):
        pred = model(x, hyperedge_index, hyperedge_type).flatten()
        loss = loss_fn(pred,y)

        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_loop(model, loss_fn, optimizer, epochs, hyperedge_index, hyperedge_type,x,y)

torch.save(model.state_dict(), './model.pt')
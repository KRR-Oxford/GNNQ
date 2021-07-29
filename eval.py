import torch
from model import HGNN
from data import data
import numpy as np
from torchmetrics import Accuracy, Precision, Recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

triples, entity2id, relation2id, id2entity, id2relation = data(['train/graph.ttl'])
with open('val/answers_val_small.npy', 'rb') as f:
    true_answers = torch.tensor(np.load(f))

num_nodes = len(entity2id)
base_dim = 16
num_layers = 3

x = torch.cat((torch.ones(num_nodes,1), torch.zeros(num_nodes,base_dim - 1)), dim=1)

# K - hot vector indicating all answers
y = torch.scatter(torch.zeros(num_nodes, dtype=torch.int16), 0, true_answers, torch.ones(num_nodes,dtype=torch.int16))

entities = torch.tensor(triples['train/graph.ttl']).t()[[0,2]]
entities = torch.cat((entities, entities[[1,0]]), dim=1)
relations = torch.tensor(triples['train/graph.ttl']).t()[1]
relations = torch.cat((relations, relations + torch.max(relations) + 1),dim=0)
hyperedge_index = {1: entities}
hyperedge_type = {1: relations}

num_edge_types_by_shape = {1:len(relation2id) * 2}

model = HGNN(base_dim, num_edge_types_by_shape, num_layers)
model.to(device)
help = model.parameters()
for param in model.parameters():
    print(type(param.data), param.size())

model.load_state_dict(torch.load('./model.pt'))

model.eval()
pred = model(x, hyperedge_index, hyperedge_type).flatten()
print(pred[true_answers])
print(pred[torch.randint(x.size()[0], (len(true_answers),))])
accuracy = Accuracy(threshold=0.5)
precision = Precision(threshold=0.5)
recall = Recall(threshold=0.5)
acc = accuracy(pred,y)
pre = precision(pred,y)
rec = recall(pred,y)


print('Accuracy ' + str(acc.item()))
print('Precision ' + str(pre.item()))
print('Recall ' + str(rec.item()))
import torch
from model import HGNN
from data import load_data
import numpy as np
import argparse
from torchmetrics import Accuracy, Precision, Recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_data', type=str, default='train')
parser.add_argument('--val_data', type=str, default='val')
parser.add_argument('--base_dim', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--val_epochs', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.1)
args = parser.parse_args()

x_val, y_val, hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape , _= load_data(args.val_data + '/graph.ttl', args.val_data + '/answers.npy', args.val_data + '/sub_query_answers_val.npy', args.base_dim)

model = HGNN(args.base_dim, num_edge_types_by_shape, args.num_layers)
model.to(device)
help = model.parameters()
for param in model.parameters():
    print(type(param.data), param.size())

model.load_state_dict(torch.load('./model.pt'))

model.eval()
pred = model(x_val, hyperedge_index_val, hyperedge_type_val).flatten()
accuracy = Accuracy(threshold=0.5)
precision = Precision(threshold=0.5)
recall = Recall(threshold=0.5)
acc = accuracy(pred,y_val)
pre = precision(pred,y_val)
rec = recall(pred,y_val)


print('Accuracy ' + str(acc.item()))
print('Precision ' + str(pre.item()))
print('Recall ' + str(rec.item()))
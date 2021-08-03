import torch
from model import HGNN
from data import data, create_y_vector, create_index_matrices, load_answers, load_subquery_answers
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

triples_val, entity2id_val, relation2id_val, _, _ = data(args.val_data + '/graph.ttl')
num_nodes_val = len(entity2id_val)
num_rel_val = len(relation2id_val)
answers = load_answers(args.val_data + '/answers.npy')
y_val = create_y_vector(answers, num_nodes_val)
subquery_answers_val = load_subquery_answers(args.val_data + '/sub_query_answers.npy')
x_val = torch.cat((torch.ones(num_nodes_val,1), torch.zeros(num_nodes_val, args.base_dim - 1)), dim=1)
hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = create_index_matrices(triples_val, num_rel_val, subquery_answers_val)

model = HGNN(args.base_dim, num_edge_types_by_shape_val, args.num_layers)
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
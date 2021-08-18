import torch
from model import HGNN
from utils import load_triples, load_answers, create_triples_with_ids, create_y_vector, create_index_matrices, add_tuples_to_index_matrices
import numpy as np
import argparse
from torchmetrics import Accuracy, Precision, Recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_data', type=str, default='train_large_2')
parser.add_argument('--val_data', type=str, default='train')
parser.add_argument('--base_dim', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=4)
args = parser.parse_args()

val_subquery_answers_files = []

triples = load_triples(args.train_data + '/graph.ttl')
_, _, relation2id, _, _ = create_triples_with_ids(triples)
triples_val = load_triples(args.val_data + '/graph.ttl')
triples_val, entity2id_val, _, id2entity_val, _ = create_triples_with_ids(triples_val, relation2id)
num_nodes_val = len(entity2id_val)
val_answers = load_answers(args.val_data + '/answers.pickle')
val_answers = [entity2id_val[entity[0]] for entity in val_answers]
y_val = create_y_vector(val_answers, num_nodes_val)
hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = create_index_matrices(triples_val)
for file in val_subquery_answers_files:
    subquery_answers_val = load_answers(args.val_data + file)
    subquery_answers_val = [[entity2id_val[entity] for entity in answer] for answer in subquery_answers_val]
    hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val = add_tuples_to_index_matrices(subquery_answers_val, hyperedge_index_val, hyperedge_type_val, num_edge_types_by_shape_val)

x_val = torch.cat((torch.ones(num_nodes_val, 1), torch.zeros(num_nodes_val, args.base_dim - 1)), dim=1)

model = HGNN(args.base_dim, num_edge_types_by_shape_val, args.num_layers)
model.to(device)
help = model.parameters()
for param in model.parameters():
    print(type(param.data), param.size())

model.load_state_dict(torch.load('./trial0.pt'))

model.eval()
pred = model(x_val, hyperedge_index_val, hyperedge_type_val).flatten()
pred_index = (pred >= 0.5).nonzero(as_tuple=True)[0].tolist()
false_positive = list(set(pred_index) - set(val_answers))
predictions = [id2entity_val[entity] for entity in false_positive]
print(predictions)
accuracy = Accuracy(threshold=0.5)
precision = Precision(threshold=0.5)
recall = Recall(threshold=0.5)
acc = accuracy(pred, torch.tensor(y_val.clone().detach(), dtype=int))
pre = precision(pred, torch.tensor(y_val.clone().detach(), dtype=int))
rec = recall(pred, torch.tensor(y_val.clone().detach(), dtype=int))

print('Accuracy ' + str(acc.item()))
print('Precision ' + str(pre.item()))
print('Recall ' + str(rec.item()))
import torch
from model import HGNN
from data import load_data
import numpy as np
import argparse
from torchmetrics import Accuracy, Precision, Recall

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_data', type=str, default='train')
parser.add_argument('--val_data', type=str, default='val')
parser.add_argument('--base_dim', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--val_epochs', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.1)
args = parser.parse_args()

base_dim = args.base_dim
num_layers = args.num_layers
epochs = args.epochs
learning_rate = args.lr
val_epochs = args.val_epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train, hyperedge_index_train, hyperedge_type_train, num_edge_types_by_shape_train, answer_indices = load_data(args.train_data + '/graph.ttl', args.train_data + '/answers.npy', args.train_data + '/sub_query_answers.npy', base_dim)
x_val, y_val, hyperedge_index_val, hyperedge_type_val, _ , _= load_data(args.val_data + '/graph.ttl', args.val_data + '/answers.npy', args.val_data + '/sub_query_answers_val.npy', base_dim)

model = HGNN(base_dim, num_edge_types_by_shape_train, num_layers)
model.to(device)
help = model.parameters()
for param in model.parameters():
    print(type(param.data), param.size())

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

accuracy = Accuracy(threshold=0.5)
precision = Precision(threshold=0.5)
recall = Recall(threshold=0.5)

def train_loop(model, loss_fn, optimizer, epochs, hyperedge_index_train, hyperedge_type_train, x_train, y_train, hyperedge_index_val, hyperedge_type_val, x_val, y_val):
    for i in range(epochs):
        model.train()
        pred = model(x_train, hyperedge_index_train, hyperedge_type_train).flatten()
        pred = torch.cat((pred[answer_indices], pred[torch.randint(x_train.size()[0], size=(len(answer_indices),))]), dim=0)
        loss = loss_fn(pred, torch.cat((torch.ones(len(answer_indices)), torch.zeros(len(answer_indices))), dim = 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % val_epochs == 0:
            model.eval()
            pred = model(x_val, hyperedge_index_val, hyperedge_type_val).flatten()
            acc = accuracy(pred, y_val)
            pre = precision(pred, y_val)
            rec = recall(pred, y_val)
            print('Val')
            print(acc)
            print(pre)
            print(rec)


train_loop(model, loss_fn, optimizer, epochs, hyperedge_index_train, hyperedge_type_train, x_train, y_train, hyperedge_index_val, hyperedge_type_val, x_val, y_val)

torch.save(model.state_dict(), './model.pt')
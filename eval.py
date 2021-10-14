import torch
import os
import pickle
import json
import argparse
import torchmetrics
from data_utils import prep_data
from model import HGNN


def compute_metrics(data, model):
    loss = 0
    accuracy = torchmetrics.Accuracy(threshold=0.5)
    precision = torchmetrics.Precision(threshold=0.5)
    recall = torchmetrics.Recall(threshold=0.5)
    model.eval()
    for data_object in data:
        pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types']).flatten()
        loss = loss + torch.nn.functional.binary_cross_entropy(pred, data_object['y'])
        accuracy(pred, data_object['y'].int())
        precision(pred, data_object['y'].int())
        recall(pred, data_object['y'].int())
    acc = accuracy.compute().item()
    pre = precision.compute().item()
    re = recall.compute().item()
    # ToDo: Figure out why this returns results below 0.5
    # print('AUC ' + str(torchmetrics.functional.auc(pred, y_val_int, reorder=True).item()))
    accuracy.reset()
    precision.reset()
    recall.reset()
    return loss, acc, pre, re


def eval(test_data_directories, query_string, model_directory, base_dim, num_layers, negative_slope, aug,
         subquery_gen_strategy, subquery_depth, max_num_subquery_vars, device, summary_writer=None):
    with open(os.path.join(model_directory, 'relation2id.pickle'), 'rb') as f:
        relation2id = pickle.load(f)

    test_data, _ = prep_data(data_directories=test_data_directories, query_string=query_string, aug=aug,
                             subquery_gen_strategy=subquery_gen_strategy, subquery_depth=subquery_depth,
                             max_num_subquery_vars=max_num_subquery_vars, relation2id=relation2id)

    model = HGNN(len(test_data[0]['x'][0]), base_dim, test_data[0]['num_edge_types_by_shape'], num_layers,
                 negative_slope)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pt')))

    _, test_acc, test_pre, test_re = compute_metrics(test_data, model)

    print('Test')
    print('Accuracy ' + str(test_acc))
    print('Precision ' + str(test_pre))
    print('Recall ' + str(test_re))
    if summary_writer:
        summary_writer.add_scalar('Precision test', test_pre)
        summary_writer.add_scalar('Recall test', test_re)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--log_directory', type=str, default='')
    parser.add_argument('--test_data', type=str, nargs='+', default=[''])
    args = parser.parse_args()

    with open(os.path.join(args.log_directory, 'config.txt'), 'r') as f:
        run_args = json.load(f)

    eval(test_data_directories=args.test_data, query_string=run_args['query_string'],
         model_directory=os.path.join(args.log_directory, 'models'), base_dim=run_args['base_dim'],
         num_layers=run_args['num_layers'], negative_slope=run_args['negative_slope'], aug=run_args['aug'],
         subquery_gen_strategy=run_args['subquery_gen_strategy'], subquery_depth=run_args['subquery_depth'],
         max_num_subquery_vars=run_args['max_num_subquery_vars'], device=device)

import torch
import os
import pickle
import json
import argparse
import torchmetrics
from model import HGNN

from data_utils import create_data_object


def test(test_data_directories, query_string, model_directory, base_dim, num_layers, negative_slope, aug, max_num_subquery_vars, device):

    with open(os.path.join(model_directory, 'relation2id.pickle'), 'rb') as f:
        relation2id = pickle.load(f)

    test_data = []
    for directory in test_data_directories:
        data_object, relation2id = create_data_object(os.path.join(directory, 'graph.nt'), os.path.join(directory, 'corrupted_graph.nt'),
                                                      query_string, aug, max_num_subquery_vars, relation2id)
        test_data.append(data_object)

    model = HGNN(len(test_data[0]['x'][0]), base_dim, test_data[0]['num_edge_types_by_shape'], num_layers)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pt')))
    test_accuracy = torchmetrics.Accuracy(threshold=0.5)
    test_precision = torchmetrics.Precision(threshold=0.5)
    test_recall = torchmetrics.Recall(threshold=0.5)

    model.eval()
    for data_object in test_data:
        pred = model(data_object['x'], data_object['hyperedge_indices'], data_object['hyperedge_types'],
                     negative_slope=negative_slope).flatten()
        # pred_index = (pred >= 0.5).nonzero(as_tuple=True)[0].tolist()
        # false_positive = list(set(pred_index) - set(val_answers))
        test_accuracy(pred, data_object['y'].int())
        test_precision(pred, data_object['y'].int())
        test_recall(pred, data_object['y'].int())

    acc = test_accuracy.compute().item()
    pre = test_precision.compute().item()
    re = test_recall.compute().item()
    print('Test')
    print('Accuracy ' + str(acc))
    print('Precision ' + str(pre))
    print('Recall ' + str(re))
    test_accuracy.reset()
    test_precision.reset()
    test_recall.reset()
    # ToDo: Figure out why this returns results below 0.5
    # print('AUC ' + str(torchmetrics.functional.auc(pred, y_val_int, reorder=True).item()))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--log_directory', type=str, default='')
    parser.add_argument('--test_data', type=str, nargs='+', default=[''])
    args = parser.parse_args()

    with open(os.path.join(args.log_directory, 'config.txt'),'r') as f:
        run_args = json.load(f)

    test(args.test_data, run_args['query_string'], os.path.join(args.log_directory, 'models'), run_args['base_dim'], run_args['num_layers'], run_args['negative_slope'], run_args['aug'], run_args['max_num_subquery_vars'], device)

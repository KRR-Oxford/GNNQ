import torch
import os
import pickle
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
                                                      query_string, base_dim, aug, max_num_subquery_vars, relation2id)
        test_data.append(data_object)

    model = HGNN(base_dim, test_data[0]['num_edge_types_by_shape'], num_layers)
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
    # ToDo: Figure out why this returns results below 0.5
    # print('AUC ' + str(torchmetrics.functional.auc(pred, y_val_int, reorder=True).item()))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_directory', type=str)
    parser.add_argument('--test_data', type=str, default=['wsdbm-data-model-2/dummy'])
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--query_string', type=str,
                        default='SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }')
    parser.add_argument('--base_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--negative_slope', type=int, default=0.1)
    parser.add_argument('--max_num_subquery_vars', type=int, default=5)
    args = parser.parse_args()

    test(args.test_data, args.query_string, args.model_directory, args.base_dim, args.num_layers, args.negative_slope, args.aug, args.max_num_subquery_vars, device)

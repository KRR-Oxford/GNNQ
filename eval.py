import torch
import os
import json
import argparse
import torchmetrics
from data_utils import prep_data
from model import HGNN


def compute_metrics(data, model, threshold=0.5):
    loss = 0
    accuracy = torchmetrics.Accuracy(threshold=threshold)
    precision = torchmetrics.Precision(threshold=threshold)
    recall = torchmetrics.Recall(threshold=threshold)
    average_precision = torchmetrics.AveragePrecision()
    unobserved_precision = torchmetrics.Precision(threshold=threshold)
    unobserved_recall = torchmetrics.Recall(threshold=threshold)
    unobserved_average_precision = torchmetrics.AveragePrecision()
    model.eval()
    for data_object in data:
        pred = model(data_object['x'], data_object['indices_dict'], data_object['shapes_dict']).flatten()
        loss = loss + torch.nn.functional.binary_cross_entropy(pred, data_object['y'])
        accuracy(pred, data_object['y'].int())
        precision(pred, data_object['y'].int())
        recall(pred, data_object['y'].int())
        average_precision(pred, data_object['y'].int())
        unobserved_precision(pred[data_object['mask_observed']], data_object['y'][data_object['mask_observed']].int())
        unobserved_recall(pred[data_object['mask_observed']], data_object['y'][data_object['mask_observed']].int())
        unobserved_average_precision(pred[data_object['mask_observed']],
                                     data_object['y'][data_object['mask_observed']].int())
    acc = accuracy.compute().item()
    pre = precision.compute().item()
    re = recall.compute().item()
    auc = average_precision.compute().item()
    unobserved_pre = unobserved_precision.compute().item()
    unobserved_re = unobserved_recall.compute().item()
    unobserved_auc = unobserved_average_precision.compute().item()
    accuracy.reset()
    precision.reset()
    recall.reset()
    average_precision.reset()
    unobserved_precision.reset()
    unobserved_recall.reset()
    unobserved_average_precision.reset()
    return loss, acc, pre, re, auc, unobserved_pre, unobserved_re, unobserved_auc


def eval(test_data_directories, query_string, model_directory, base_dim, num_layers, negative_slope, aug,
         subquery_gen_strategy, subquery_depth, max_num_subquery_vars, device, summary_writer=None):
    test_data = prep_data(data_directories=test_data_directories, query_string=query_string, aug=aug,
                          subquery_gen_strategy=subquery_gen_strategy, subquery_depth=subquery_depth,
                          max_num_subquery_vars=max_num_subquery_vars)

    model = HGNN(len(test_data[0]['x'][0]), base_dim, test_data[0]['shapes_dict'], num_layers,
                 negative_slope)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pt')))

    _, test_acc, test_pre, test_re, test_auc, test_unobserved_pre, test_unobserved_re, test_unobserved_auc = compute_metrics(
        test_data, model)

    print('Testing!')
    print('Accuracy for all answers: ' + str(test_acc))
    print('Precision for all answers: ' + str(test_pre))
    print('Recall for all answers: ' + str(test_re))
    print('AUC for all answers: ' + str(test_auc))
    print('Precision for unobserved answers: ' + str(test_unobserved_pre))
    print('Recall for unobserved answers: ' + str(test_unobserved_re))
    print('AUC for unobserved answers: ' + str(test_unobserved_auc))
    if summary_writer:
        summary_writer.add_scalar('Precision for all answers on the test datasets.', test_pre)
        summary_writer.add_scalar('Recall for all answers on the test datasets.', test_re)
        summary_writer.add_scalar('AUC for all answers on the test datasets.', test_auc)
        summary_writer.add_scalar('Precision for unobserved answers on the test datasets.', test_unobserved_pre)
        summary_writer.add_scalar('Recall for unobserved answers on the test datasets.', test_unobserved_re)
        summary_writer.add_scalar('AUC for unobserved answers on the test datasets.', test_unobserved_auc)


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

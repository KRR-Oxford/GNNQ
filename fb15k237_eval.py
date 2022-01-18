import torch
import os
import json
import argparse
import torchmetrics
import pickle
from fb15k237_data_utils import prep_data


def compute_metrics(data, model, threshold=0.5):
    loss = 0
    accuracy = torchmetrics.Accuracy(threshold=threshold)
    precision = torchmetrics.Precision(threshold=threshold)
    recall = torchmetrics.Recall(threshold=threshold)
    average_precision = torchmetrics.AveragePrecision()
    model.eval()
    counter = 1
    for data_object in data:
        print('Tested on samples ({0}/{1})!'.format(counter, len(data)))
        pred = model(data_object['x'], data_object['indices_dict']).flatten()
        pred = pred[data_object['answers']]
        y = data_object['labels']
        loss = loss + torch.nn.functional.binary_cross_entropy(pred,y)
        accuracy(pred, data_object['labels'].int())
        precision(pred, data_object['labels'].int())
        recall(pred, data_object['labels'].int())
        average_precision(pred, data_object['labels'].int())
        counter += 1
    acc = accuracy.compute().item()
    pre = precision.compute().item()
    re = recall.compute().item()
    auc = average_precision.compute().item()
    accuracy.reset()
    precision.reset()
    recall.reset()
    average_precision.reset()
    return loss, acc, pre, re, auc


def eval(test_data, model_directory, aug, device, summary_writer=None):
    model = torch.load(os.path.join(model_directory, 'model.pt'))
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    infile = open(test_data, 'rb')
    test_pos_samples, test_pos_answers, test_neg_samples, test_neg_answers = pickle.load(infile)
    infile.close()

    print('Positive testing samples!')
    test_pos_data_objects = prep_data(1, test_pos_samples, test_pos_answers, aug=aug,
                                      subqueries=model.subqueries)
    print('Negative testing samples!')
    test_neg_data_objects = prep_data(0, test_neg_samples, test_neg_answers, aug=aug,
                                      subqueries=model.subqueries)

    _, test_acc, test_pre, test_re, test_auc = compute_metrics(test_pos_data_objects+test_neg_data_objects, model)

    print('Testing!')
    print('Accuracy for all answers: ' + str(test_acc))
    print('Precision for all answers: ' + str(test_pre))
    print('Recall for all answers: ' + str(test_re))
    print('AUC for all answers: ' + str(test_auc))
    if summary_writer:
        summary_writer.add_scalar('Precision for all answers on the test datasets.', test_pre)
        summary_writer.add_scalar('Recall for all answers on the test datasets.', test_re)
        summary_writer.add_scalar('AUC for all answers on the test datasets.', test_auc)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_directory', type=str, default='')
    parser.add_argument('--test_data', type=str, default='')
    args = parser.parse_args()

    with open(os.path.join(args.log_directory, 'config.txt'), 'r') as f:
        run_args = json.load(f)

    eval(test_data=args.test_data, model_directory=os.path.join(args.log_directory, 'models'),
         aug=run_args['aug'], device=device)
import torch
import os
import json
import argparse
import torchmetrics
from data_utils import prep_data
from load_watdiv import load_watdiv_benchmark
from load_fb15k237 import load_fb15k237_benchmark


def compute_metrics(data, model, threshold=0.5):
    loss = 0
    precision = torchmetrics.Precision(threshold=threshold)
    recall = torchmetrics.Recall(threshold=threshold)
    average_precision = torchmetrics.AveragePrecision()
    unobserved_precision = torchmetrics.Precision(threshold=threshold)
    unobserved_recall = torchmetrics.Recall(threshold=threshold)
    unobserved_average_precision = torchmetrics.AveragePrecision()
    model.eval()
    counter = 1
    for data_object in data:
        pred = model(data_object['feat'], data_object['indices_dict']).flatten()
        pred = pred[data_object['nodes']]
        y = data_object['labels']
        loss = loss + torch.nn.functional.binary_cross_entropy(pred, y)
        precision(pred, y.int())
        recall(pred, y.int())
        average_precision(pred, y.int())
        unobserved_precision(pred[data_object['mask']], y[data_object['mask']].int())
        unobserved_recall(pred[data_object['mask']], y[data_object['mask']].int())
        unobserved_average_precision(pred[data_object['mask']], y[data_object['mask']].int())
        print('Processed {0}/{1} samples!'.format(counter, len(data)))
        counter += 1
    pre = precision.compute().item()
    re = recall.compute().item()
    ap = average_precision.compute().item()
    unobserved_pre = unobserved_precision.compute().item()
    unobserved_re = unobserved_recall.compute().item()
    unobserved_ap = unobserved_average_precision.compute().item()
    precision.reset()
    recall.reset()
    average_precision.reset()
    unobserved_precision.reset()
    unobserved_recall.reset()
    unobserved_average_precision.reset()
    return loss, pre, re, ap, unobserved_pre, unobserved_re, unobserved_ap


def eval(test_data, model_directory, aug, device, summary_writer=None):
    model = torch.load(model_directory)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    if 'fb15k237' in test_data[0]:
        test_samples, test_answers, test_labels, mask_observed, graphs = load_fb15k237_benchmark(test_data[0])
    else:
        test_samples, test_answers, test_labels, mask_observed, graphs = load_watdiv_benchmark(test_data,
                                                                                               model.query_string)

    test_data_objects = prep_data(test_labels, test_samples, test_answers, mask_observed, aug=aug,
                                  subqueries=model.subqueries)

    _, test_pre, test_re, test_ap, test_unobserved_pre, test_unobserved_re, test_unobserved_ap = compute_metrics(
        test_data_objects, model)

    print('Testing!')
    print('Precision for all samples: ' + str(test_pre))
    print('Recall for all samples: ' + str(test_re))
    print('AP for all samples: ' + str(test_ap))
    print('Precision for unmasked samples: ' + str(test_unobserved_pre))
    print('Recall for unmasked samples: ' + str(test_unobserved_re))
    print('AP for unmasked samples: ' + str(test_unobserved_ap))
    if summary_writer:
        summary_writer.add_scalar('Precision for all testing samples.', test_pre)
        summary_writer.add_scalar('Recall for all testing samples.', test_re)
        summary_writer.add_scalar('AP for all testing samples.', test_ap)
        summary_writer.add_scalar('Precision for unmasked testing samples.', test_unobserved_pre)
        summary_writer.add_scalar('Recall for unmasked testing samples.', test_unobserved_re)
        summary_writer.add_scalar('AP for unmasked testing samples.', test_unobserved_ap)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str)
    parser.add_argument('--test_data', type=str, nargs='+')
    args = parser.parse_args()

    with open(os.path.join(args.log_directory, 'config.txt'), 'r') as f:
        run_args = json.load(f)

    eval(test_data=args.test_data, model_directory=args.model_directory,
         aug=run_args['aug'], device=device)

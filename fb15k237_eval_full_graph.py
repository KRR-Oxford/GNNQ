import torch
import os
import argparse
import torchmetrics
from rdflib import Graph
from data_utils import create_data_object
from create_samples_from_kg import create_witness_graphs, ground_rule, corrupt_graph, create_rules_dict
from load_fb15k237 import load_fb15k237_benchmark
import re
import json

def create_bgp_query_from_monadic_query(query_string):
    bgp = re.search(r'\{(.*?)\}',query_string).group(1)
    vars = re.findall(r'\?\w*\b', bgp)
    vars = ' '.join(vars)
    return re.sub(r'(?<=select )(.*?)(?= where)',vars, query_string)

def create_sample(g, query, answer, witness_graphs, positive, completion_rules):
    accept = False
    while not accept:
        sample_graph = Graph()
        for triple in g:
            sample_graph.add(triple)
        sample_graph = corrupt_graph(sample_graph, 0.2)
        triples_can_be_recovered = Graph()
        witnesses_for_answer = witness_graphs[answer]
        witness_triples_for_answer = Graph()
        for witness_for_answer, _ in witnesses_for_answer:
            for triple in witness_for_answer:
                witness_triples_for_answer.add(triple)
        sample_graph = sample_graph - witness_triples_for_answer
        if positive:
            # Corrupt witness such that it can be recovered with completion function -> positive sample
            for triple in witness_triples_for_answer:
                if torch.bernoulli(torch.tensor([0.8])) or (str(triple[1]) not in completion_rules.keys()):
                    if torch.bernoulli(torch.tensor([0.5])) and (str(triple[1]) in completion_rules.keys()):
                        ground_rule(witness_triples_for_answer, triple, completion_rules)
                else:
                    witness_triples_for_answer.remove(triple)
                    ground_rule(witness_triples_for_answer, triple, completion_rules)
            qres = witness_triples_for_answer.query(query)
            print(len(qres))
            if len(qres) == 0:
                print('Positive sample accepted!')
                accept = True
            else:
                print('Positive sample rejected!')
        else:
            # Corrupt witness such that it can not be recovered with completion function -> negative sample
            triples_can_be_recovered = Graph()
            for triple in witness_triples_for_answer:
                if torch.bernoulli(torch.tensor([0.8])) or (str(triple[1]) not in completion_rules.keys()):
                    if torch.bernoulli(torch.tensor([0.75])) and (str(triple[1]) in completion_rules.keys()):
                        ground_rule(witness_triples_for_answer, triple, completion_rules)
                        # Are later removed and can be recovered
                        if torch.bernoulli(torch.tensor([0.5])):
                            triples_can_be_recovered.add(triple)
                else:
                    witness_triples_for_answer.remove(triple)
            qres = witness_triples_for_answer.query(query)
            print(len(qres))
            if len(qres) == 0:
                print('Negative sample accepted!')
                accept = True
            else:
                print('Negative sample rejected!')
    graph = (sample_graph + witness_triples_for_answer) - triples_can_be_recovered
    print(len(graph))
    return graph



def eval(g, answers,  aug, model, summary_writer=None, threshold=0.5):
    with torch.no_grad():
        model.to(device)
        model.eval()
        completion_rules = create_rules_dict()
        for param in model.parameters():
            print(type(param.data), param.size())
        accuracy = torchmetrics.Accuracy(threshold=threshold)
        precision = torchmetrics.Precision(threshold=threshold)
        recall = torchmetrics.Recall(threshold=threshold)
        average_precision = torchmetrics.AveragePrecision()
        bgp = create_bgp_query_from_monadic_query(model.query_string)
        witness_graphs, _ = create_witness_graphs(g, bgp)
        counter = 1
        for answer in answers:
            print('Computed ({0}/{1}) samples!'.format(counter, len(answers)))
            label = torch.bernoulli(torch.tensor([0.5]))
            sample = create_sample(g, model.query_string, answer[0], witness_graphs, label, completion_rules)
            data_object = create_data_object([label], sample, answer, [True], aug, model.subqueries)
            if data_object == None:
                counter += 1
                continue
            pred = model(data_object['feat'], data_object['indices_dict']).flatten()
            pred = pred[data_object['nodes']]
            y = data_object['labels'].int()
            accuracy(pred, y)
            precision(pred, y)
            recall(pred, y)
            average_precision(pred, y)
            acc = accuracy.compute().item()
            pre = precision.compute().item()
            re = recall.compute().item()
            auc = average_precision.compute().item()

            print('Current accuracy: ' + str(acc))
            print('Current precision: ' + str(pre))
            print('Current recall: ' + str(re))
            print('Current AP: ' + str(auc))
            counter += 1


        acc = accuracy.compute().item()
        pre = precision.compute().item()
        re = recall.compute().item()
        auc = average_precision.compute().item()
        accuracy.reset()
        precision.reset()
        recall.reset()
        average_precision.reset()

        print('Testing!')
        print('Accuracy for all samples: ' + str(acc))
        print('Precision for all samples: ' + str(pre))
        print('Recall for all samples: ' + str(re))
        print('AP for all samples: ' + str(auc))
        if summary_writer:
            summary_writer.add_scalar('Precision for all test samples', pre)
            summary_writer.add_scalar('Recall for all test samples.', re)
            summary_writer.add_scalar('AP for all test samples.', auc)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str)
    parser.add_argument('--log_directory', type=str, default='')
    parser.add_argument('--test_data', type=str, nargs='+')
    args = parser.parse_args()

    g = Graph()
    g.parse(args.graph, format="nt")

    with open(os.path.join(args.log_directory, 'config.txt'), 'r') as f:
        run_args = json.load(f)

    test_samples, test_answers, test_labels, mask_observed, graphs = load_fb15k237_benchmark(args.test_data[0])

    model = torch.load(os.path.join(args.log_directory, 'models/model.pt'))

    eval(g, test_answers, run_args['aug'], model)
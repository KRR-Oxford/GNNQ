import torch
import os
import argparse
import torchmetrics
import pickle
from rdflib import Graph
from fb15k237_data_utils import create_data_object
from create_samples_from_kg import create_witness_graphs, ground_rule, corrupt_graph, rules


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



def eval(g, query, bgp, answers,  aug, model_directory, summary_writer=None, threshold=0.5):
    with torch.no_grad():
        model = torch.load(os.path.join(model_directory, 'model.pt'))
        model.to(device)
        model.eval()
        completion_rules = rules()
        for param in model.parameters():
            print(type(param.data), param.size())
        accuracy = torchmetrics.Accuracy(threshold=threshold)
        precision = torchmetrics.Precision(threshold=threshold)
        recall = torchmetrics.Recall(threshold=threshold)
        average_precision = torchmetrics.AveragePrecision()
        witness_graphs, _ = create_witness_graphs(g, bgp)
        counter = 1
        for answer in answers:
            print('Computed ({0}/{1}) samples!'.format(counter, len(answers)))
            label = torch.bernoulli(torch.tensor([0.5]))
            sample = create_sample(g, query, answer, witness_graphs, label, completion_rules)
            data_object = create_data_object([label], sample, [answer], aug, model.subqueries)
            if data_object == None:
                counter += 1
                continue
            pred = model(data_object['x'], data_object['indices_dict']).flatten()
            pred = pred[data_object['answers']]
            accuracy(pred, data_object['labels'].int())
            precision(pred, data_object['labels'].int())
            recall(pred, data_object['labels'].int())
            average_precision(pred, data_object['labels'].int())
            # acc = accuracy.compute().item()
            # pre = precision.compute().item()
            # re = recall.compute().item()
            # auc = average_precision.compute().item()
            #
            # print('Accuracy for all answers: ' + str(acc))
            # print('Precision for all answers: ' + str(pre))
            # print('Recall for all answers: ' + str(re))
            # print('AUC for all answers: ' + str(auc))
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
        print('Accuracy for all answers: ' + str(acc))
        print('Precision for all answers: ' + str(pre))
        print('Recall for all answers: ' + str(re))
        print('AUC for all answers: ' + str(auc))
        if summary_writer:
            summary_writer.add_scalar('Precision for all answers on the test datasets.', pre)
            summary_writer.add_scalar('Recall for all answers on the test datasets.', re)
            summary_writer.add_scalar('AUC for all answers on the test datasets.', auc)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--graph', type=str)
    parser.add_argument('--query', type=str, default='')
    parser.add_argument('--bgp', type=str, default='')
    parser.add_argument('--log_directory', type=str, default='')
    parser.add_argument('--test_data', type=str, default='')
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args()

    g = Graph()
    g.parse(args.graph, format="nt")

    infile = open(args.test_data, 'rb')
    test_pos_samples, test_pos_answers, test_neg_samples, test_neg_answers = pickle.load(infile)
    infile.close()


    eval(g, args.query, args.bgp, test_pos_answers+test_neg_answers, args.aug, args.log_directory)
from rdflib import Graph, URIRef
from rdflib.plugins.sparql import prepareQuery
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph, rdflib_to_networkx_multidigraph
import networkx as nx
import random
import itertools
from collections import defaultdict
import uuid
import torch
import copy
import pickle
import argparse


def randomly_split_list(l):
    ones = int(len(l) / 2)
    zeros = len(l) - ones
    mask = [True] * ones + [False] * zeros
    random.shuffle(mask)
    inversed_mask = [not b00l for b00l in mask]
    pos = list(itertools.compress(l, mask))
    neg = list(itertools.compress(l, inversed_mask))
    return pos, neg, mask, inversed_mask


def extract_connected_subgraph_of_khop(g, start_nodes, k=2, current_iter=0):
    nodes = set()
    for entity in start_nodes:
        for edge in nx.bfs_edges(g, source=entity, depth_limit=1):
            if torch.bernoulli(torch.tensor([0.01])):
                nodes.add(edge[1])
    if current_iter < k:
        return start_nodes.union(extract_connected_subgraph_of_khop(g, start_nodes.union(nodes), k, current_iter + 1))
    else:
        return start_nodes.union(nodes)


def networkx_multidigraph_to_rdflib(networkx_graph):
    rdflib_graph = Graph()
    for edge in networkx_graph.edges:
        rdflib_graph.add((edge[0], edge[2], edge[1]))
    return rdflib_graph


def rules(rule_file='./rules.txt'):
    input_file = open(rule_file, "r")
    lines = input_file.readlines()
    rules_by_head_predicate = defaultdict(list)
    for line in lines:
        split = line.split('\t')
        rule = split[3]
        atoms = rule.split(' ')
        head_predicate = atoms[0].split('(')[0]
        head_predicate = "http://dummyrel.com" + head_predicate
        rules_by_head_predicate[head_predicate].append(rule)
    return dict(rules_by_head_predicate)


def ground_rule(g, triple, rules):
    head = str(triple[1])
    rule = random.choice(rules[head])
    atoms = rule.split(' ')
    new_object = URIRef("http://dummy.com" + str(uuid.uuid4()))
    g.add((triple[0], URIRef("http://dummyrel.com" + atoms[2].split('(')[0]), new_object))
    new_subject = new_object
    for atom in atoms[3:-1]:
        new_object = URIRef("http://dummy.com" + str(uuid.uuid4()))
        g.add((new_subject, URIRef("http://dummyrel.com" + atom.split('(')[0]), new_object))
        new_subject = new_object
    g.add((new_subject, URIRef("http://dummyrel.com" + atoms[-1].split('(')[0]), triple[2]))

def corrupt_graph(g,prob):
    for triple in g:
        if torch.bernoulli(torch.tensor([prob])):
            g.remove(triple)
    return g


def create_test_sample_graphs(g, query, answers, witness_graphs, completion_rules, positive):
    samples = []
    used_answers = []
    for answer in answers:
        sample_graph = copy.deepcopy(g)
        sample_graph = corrupt_graph(sample_graph, 0.2)
        witnesses_for_answer = witness_graphs[answer]
        witness_triples_for_answer = Graph()
        for witness_for_answer, _ in witnesses_for_answer:
            for triple in witness_for_answer:
                witness_triples_for_answer.add(triple)
        sample_graph = sample_graph - witness_triples_for_answer
        qres = witness_triples_for_answer.query(query)
        print(len(qres))
        accept = False
        while not accept:
            if positive:
                # Corrupt witness such that it can be recovered with completion function -> positive sample
                print('Positive sample!')
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
                    samples.append(sample_graph + witness_triples_for_answer)
                    used_answers.append(answer)
                    accept = True
                else:
                    print('Positive sample rejected!')
            else:
                # Corrupt witness such that it can not be recovered with completion function -> negative sample
                print('Negative sample!')
                dummy = Graph()
                for triple in witness_triples_for_answer:
                    if torch.bernoulli(torch.tensor([0.8])) or (str(triple[1]) not in completion_rules.keys()):
                        if torch.bernoulli(torch.tensor([0.75])) and (str(triple[1]) in completion_rules.keys()):
                            ground_rule(witness_triples_for_answer, triple, completion_rules)
                            # Are later removed and can be recovered
                            if torch.bernoulli(torch.tensor([0.5])):
                                dummy.add(triple)
                    else:
                        witness_triples_for_answer.remove(triple)
                qres = witness_triples_for_answer.query(query)
                print(len(qres))
                if len(qres) == 0:
                    print('Negative sample accepted!')
                    samples.append((sample_graph + witness_triples_for_answer) - dummy)
                    used_answers.append(answer)
                    accept = True
                else:
                    print('Negative sample rejected!')
    return samples, used_answers



def create_samples_graphs(g, answers, witness_graphs, witness_triples, completion_rules, positive):
    g_no_witnesses = g - witness_triples
    nx_g_no_witnesses = rdflib_to_networkx_graph(g_no_witnesses)
    samples = []
    used_answers = []
    for answer in answers:
        witness_graph, witness_entities = random.choice(witness_graphs[answer])
        # check that all entities occur in the cleaned KG
        if not all(e in nx_g_no_witnesses.nodes for e in witness_entities):
            print('Sample entity not contained in KG!')
            continue
        # Sample from cleaned_q + witness?
        sample_entities = extract_connected_subgraph_of_khop(nx_g_no_witnesses, witness_entities, 2)
        print('Sample entities')
        print(len(sample_entities))
        sample_graph = rdflib_to_networkx_multidigraph(g_no_witnesses).subgraph(list(sample_entities)).copy()
        print('Sample graph size')
        print(sample_graph.number_of_edges())
        sample_graph = networkx_multidigraph_to_rdflib(sample_graph)
        accept = False
        while not accept:
            witness_graph_copy = copy.deepcopy(witness_graph)
            if positive:
                # Corrupt witness such that it can be recovered with completion function -> positive sample
                print('Positive sample!')
                for triple in witness_graph:
                    if torch.bernoulli(torch.tensor([0.9])) or (str(triple[1]) not in completion_rules.keys()):
                        if torch.bernoulli(torch.tensor([0.5])) and (str(triple[1]) in completion_rules.keys()):
                            ground_rule(witness_graph_copy, triple, completion_rules)
                    else:
                        witness_graph_copy.remove(triple)
                        print('Corrupted sample!')
                        ground_rule(witness_graph_copy, triple, completion_rules)
                        accept = True
                if accept:
                    print('Positive sample accepted!')
                    samples.append(sample_graph + witness_graph_copy)
                    used_answers.append(answer)
                else:
                    print('Positive sample rejected!')
            else:
                # Corrupt witness such that it can not be recovered with completion function -> negative sample
                print('Negative sample!')
                for triple in witness_graph:
                    if torch.bernoulli(torch.tensor([0.8])) or (str(triple[1]) not in completion_rules.keys()):
                        if torch.bernoulli(torch.tensor([0.5])) and (str(triple[1]) in completion_rules.keys()):
                            ground_rule(witness_graph_copy, triple, completion_rules)
                    else:
                        witness_graph_copy.remove(triple)
                        if torch.bernoulli(torch.tensor([0.5])):
                            ground_rule(witness_graph_copy, triple, completion_rules)
                        else:
                            print('Cannot be recovered!')
                            accept = True
                if accept:
                    print('Negative sample accepted!')
                    samples.append(sample_graph + witness_graph_copy)
                    used_answers.append(answer)
                else:
                    print('Negative sample rejected!')
    return samples, used_answers


def create_samples(g, query, train_file, val_file, test_file, samples_per_answer):
    completion_rules = rules()
    q = prepareQuery(query)
    qres = g.query(q)
    witness_graphs = defaultdict(list)
    witness_triples = Graph()
    print('BGP matches!')
    print(len(qres))
    for row in qres.bindings:
        dummy = Graph()
        sample_entities = set()
        for triple in q.algebra['p']['p']['triples']:
            dummy.add((row[triple[0]], triple[1], row[triple[2]]))
            witness_triples.add((row[triple[0]], triple[1], row[triple[2]]))
            sample_entities.add(row[triple[0]])
            sample_entities.add(row[triple[2]])
        witness_graphs[row[q.algebra['PV'][0]]].append((dummy, sample_entities))
    print('Constructed witness graphs!')
    answers = list(witness_graphs.keys())
    train_answers, test_answers, _, _ = randomly_split_list(answers)
    val_answers, test_answers, _, _ = randomly_split_list(test_answers)

    train_pos_answers, train_neg_answers, _, _ = randomly_split_list(train_answers)
    val_pos_answers, val_neg_answers, _, _ = randomly_split_list(val_answers)
    test_pos_answers, test_neg_answers, _, _ = randomly_split_list(test_answers)

    train_pos_answers = list(itertools.chain.from_iterable(itertools.repeat(x, samples_per_answer) for x in train_pos_answers))
    train_neg_answers = list(itertools.chain.from_iterable(itertools.repeat(x, samples_per_answer) for x in train_neg_answers))

    val_pos_answers = list(itertools.chain.from_iterable(itertools.repeat(x, samples_per_answer) for x in val_pos_answers))
    val_neg_answers = list(itertools.chain.from_iterable(itertools.repeat(x, samples_per_answer) for x in val_neg_answers))

    test_pos_answers = list(itertools.chain.from_iterable(itertools.repeat(x, samples_per_answer) for x in test_pos_answers))
    test_neg_answers = list(itertools.chain.from_iterable(itertools.repeat(x, samples_per_answer) for x in test_neg_answers))

    # Train samples!
    train_pos_samples, train_pos_answers = create_samples_graphs(g, train_pos_answers, witness_graphs,
                                                   witness_triples, completion_rules, True)
    train_neg_samples, train_neg_answers = create_samples_graphs(g, train_neg_answers, witness_graphs,
                                                   witness_triples, completion_rules, False)
    outfile = open(train_file, 'wb')
    pickle.dump((train_pos_samples, train_pos_answers, train_neg_samples, train_neg_answers), outfile)
    outfile.close()

    # Val samples!
    val_pos_samples, val_pos_answers = create_samples_graphs(g, val_pos_answers, witness_graphs,
                                                                 witness_triples, completion_rules, True)
    val_neg_samples, val_neg_answers = create_samples_graphs(g, val_neg_answers, witness_graphs,
                                                                 witness_triples, completion_rules, False)
    outfile = open(val_file, 'wb')
    pickle.dump((val_pos_samples, val_pos_answers, val_neg_samples, val_neg_answers), outfile)
    outfile.close()

    # Test samples!
    test_pos_samples, test_pos_answers = create_samples_graphs(g, test_pos_answers, witness_graphs,
                                              witness_triples, completion_rules, True)
    test_neg_samples, test_neg_answers = create_samples_graphs(g, test_neg_answers, witness_graphs,
                                              witness_triples, completion_rules, False)
    outfile = open(test_file, 'wb')
    pickle.dump((test_pos_samples, test_pos_answers, test_neg_samples, test_neg_answers), outfile)
    outfile.close()

    test_pos_samples_full_graph, test_pos_answers_full_graph = create_test_sample_graphs(g, query, test_pos_answers, witness_graphs, completion_rules, True)
    test_neg_samples_full_graph, test_neg_answers_full_graph = create_test_sample_graphs(g, query, test_neg_answers, witness_graphs, completion_rules, False)
    outfile = open('full_graph' + test_file, 'wb')
    pickle.dump((test_pos_samples_full_graph, test_pos_answers_full_graph, test_neg_samples_full_graph, test_neg_answers_full_graph), outfile)
    outfile.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_string', type=str)
    parser.add_argument('--graph', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--samples_per_answer', type=int)
    args = parser.parse_args()
    graph = args.graph

    g = Graph()
    g.parse(graph, format="nt")

    create_samples(g, args.query_string, args.train_file, args.val_file, args.test_file, args.samples_per_answer)
    print('Done')

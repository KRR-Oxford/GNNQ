import os
import torch
import pickle
from rdflib import Graph


def compute_query_answers(graph, query_string):
    qres = graph.query(query_string)
    answers = []
    for row in qres:
        answers.append([str(entity).strip() for entity in row][0])
    return answers

def get_all_nodes(graph):
    nodes = set()
    for s, p, o in graph:
        nodes.add(str(s).strip())
        if not str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            nodes.add(str(o).strip())
    return list(nodes)

def get_all_types():
    with open('datasets/watdiv/types_dict.pickle', 'rb') as f:
        types = pickle.load(f)
    return types

def load_watdiv_benchmark(directories, query_string):
    incomplete_graphs = []
    complete_graphs = []
    nodes = []
    labels = []
    masks = []
    types = get_all_types()
    total = 0
    total_observed = 0
    total_unobserved = 0
    for directory in directories:
        print('Preparing dataset: ' + directory)
        g = Graph(store="Oxigraph")
        g.parse(os.path.join(directory, 'graph.nt'), format="nt")
        corrupted_g = Graph(store="Oxigraph")
        corrupted_g.parse(os.path.join(directory, 'corrupted_graph.nt'), format="nt")
        pos_nodes = compute_query_answers(g, query_string)
        print("Answers-{}".format(len(pos_nodes)))
        total = total + len(pos_nodes)
        observed_nodes = compute_query_answers(corrupted_g, query_string)
        print("Observed answers-{}".format(len(observed_nodes)))
        total_observed = total_observed + len(observed_nodes)
        print("Unobserved answers-{}".format(len(pos_nodes) - len(observed_nodes)))
        total_unobserved = total_unobserved + (len(pos_nodes) - len(observed_nodes))
        graph_nodes = get_all_nodes(corrupted_g)
        neg_nodes = [e for e in graph_nodes if e not in pos_nodes]
        sample_nodes = pos_nodes + neg_nodes
        mask = [False if e in observed_nodes else True for e in sample_nodes]
        incomplete_graphs.append(corrupted_g)
        complete_graphs.append(g)
        nodes.append(pos_nodes + neg_nodes)
        labels.append(torch.cat((torch.ones(len(pos_nodes)), torch.zeros(len(neg_nodes))), dim=0))
        masks.append(mask)
    print("Total answers-{}".format(total))
    print("Total observed answers-{}".format(total_observed))
    print("Total unobserved answers-{}".format(total_unobserved))
    return incomplete_graphs, nodes, types, labels, masks, complete_graphs

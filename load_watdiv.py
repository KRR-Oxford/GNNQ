import os
import torch
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
        nodes.add(str(o).strip())
    return list(nodes)


def load_watdiv_benchmark(directories, query_string):
    incomplete_graphs = []
    complete_graphs = []
    nodes = []
    labels = []
    masks = []
    for directory in directories:
        print('Preparing dataset: ' + directory)
        g = Graph()
        g.parse(os.path.join(directory, 'graph.nt'), format="nt")
        corrupted_g = Graph()
        corrupted_g.parse(os.path.join(directory, 'corrupted_graph.nt'), format="nt")
        pos_nodes = compute_query_answers(g, query_string)
        observed_nodes = compute_query_answers(corrupted_g, query_string)
        graph_nodes = get_all_nodes(corrupted_g)
        neg_nodes = [e for e in graph_nodes if e not in pos_nodes]
        sample_nodes = pos_nodes + neg_nodes
        mask = [False if e in observed_nodes else True for e in sample_nodes]
        incomplete_graphs.append(corrupted_g)
        complete_graphs.append(g)
        nodes.append(pos_nodes + neg_nodes)
        labels.append(torch.cat((torch.ones(len(pos_nodes)), torch.zeros(len(neg_nodes))), dim=0))
        masks.append(mask)
    return incomplete_graphs, nodes, labels, masks, complete_graphs

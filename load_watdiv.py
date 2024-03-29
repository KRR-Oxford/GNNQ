import os
import torch
import pickle
import re
from rdflib import Graph


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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

def load_watdiv_benchmark(directory, query_string):
    incomplete_graphs = []
    complete_graphs = []
    nodes = []
    labels = []
    masks = []
    types = get_all_types()
    total = 0
    total_neg = 0
    total_observed = 0
    total_unobserved = 0
    subdirectories = os.listdir(directory)
    subdirectories.sort(key=natural_keys)
    for dataset_directory in subdirectories:
        dataset_directory = os.path.join(directory, dataset_directory)
        if os.path.isdir(dataset_directory):
            print('Preparing dataset: ' + dataset_directory)
            g = Graph(store="Oxigraph")
            g.parse(os.path.join(dataset_directory, 'graph.nt'), format="nt")
            corrupted_g = Graph(store="Oxigraph")
            corrupted_g.parse(os.path.join(dataset_directory, 'corrupted_graph.nt'), format="nt")
            pos_nodes = compute_query_answers(g, query_string)
            print("Pos nodes-{}".format(len(pos_nodes)))
            total = total + len(pos_nodes)
            observed_nodes = compute_query_answers(corrupted_g, query_string)
            print("Observed pos nodes-{}".format(len(observed_nodes)))
            total_observed = total_observed + len(observed_nodes)
            print("Unobserved pos nodes-{}".format(len(pos_nodes) - len(observed_nodes)))
            total_unobserved = total_unobserved + (len(pos_nodes) - len(observed_nodes))
            graph_nodes = get_all_nodes(corrupted_g)
            neg_nodes = [e for e in graph_nodes if e not in pos_nodes]
            print("Neg nodes-{}".format(len(neg_nodes)))
            total_neg = total_neg + len(neg_nodes)
            sample_nodes = pos_nodes + neg_nodes
            mask = [False if e in observed_nodes else True for e in sample_nodes]
            incomplete_graphs.append(corrupted_g)
            complete_graphs.append(g)
            nodes.append(pos_nodes + neg_nodes)
            labels.append(torch.cat((torch.ones(len(pos_nodes)), torch.zeros(len(neg_nodes))), dim=0))
            masks.append(mask)
    print("Total pos nodes-{}".format(total))
    print("Total observed pos nodes-{}".format(total_observed))
    print("Total unobserved pos nodes-{}".format(total_unobserved))
    print("Total neg nodes-{}".format(total_neg))
    return incomplete_graphs, nodes, types, labels, masks, complete_graphs

import torch
from rdflib import Graph, URIRef
from argparse import ArgumentParser


def delete_edges_randomly(data_directory, drop_prop):
    g = Graph()
    g.parse(data_directory + '/graph.nt', format="nt")
    for s, p, o in g:
        drop = torch.bernoulli(p=drop_prop, input=torch.tensor([0])).item() == 1
        if drop:
            g.remove((s, p, o))
    g.serialize(destination=data_directory + '/corrupted_graph.nt', format='nt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../datasets/fb15k237_rules_mat/dataset3')
    parser.add_argument('--drop_prop', type=int, default=0.2)
    args = parser.parse_args()
    delete_edges_randomly(args.data_dir, args.drop_prop)
    print('Done')

from argparse import ArgumentParser
from collections import defaultdict
from rdflib import Graph, URIRef
import torch
import os
import uuid

parser = ArgumentParser()
parser.add_argument('--data_dir', help="File with all the rules", default='../datasets/fb15k237rules/dataset2')
parser.add_argument('--rules_dir', help="File with all the rules",
                    default='../datasets/fb15k237rules/rules/gamma-1000_filtered.txt')
parser.add_argument('--drop_prop', type=float, default=0.20)
args = parser.parse_args()

input_file = open(args.rules_dir, "r")
lines = input_file.readlines()

rules_by_head_predicate = defaultdict(list)
for line in lines:
    split = line.split('\t')
    rule = split[3]
    atoms = rule.split(' ')
    head_predicate = atoms[0].split('(')[0]
    head_predicate = "http://dummyrel.com" + head_predicate
    rules_by_head_predicate[head_predicate].append(rule)

g = Graph()
g.parse(os.path.join(args.data_dir, 'graph.nt'), format="nt")
dummy_url = "http://dummyurlval.com/"
for s, p, o in g:
    if str(p) in rules_by_head_predicate.keys():
        i = torch.randint(len(rules_by_head_predicate[str(p)]), (1,))
        rule = rules_by_head_predicate[str(p)][i]
        atoms = rule.split(' ')
        new_object = URIRef(dummy_url + str(uuid.uuid4()))
        g.add((s, URIRef("http://dummyrel.com" + atoms[2].split('(')[0]), new_object))
        new_subject = new_object
        for atom in atoms[3:-1]:
            new_object = URIRef(dummy_url + str(uuid.uuid4()))
            g.add((new_subject, URIRef("http://dummyrel.com" + atom.split('(')[0]), new_object))
            new_subject = new_object
        g.add((new_subject, URIRef("http://dummyrel.com" + atoms[-1].split('(')[0]), o))

        drop = torch.bernoulli(p=args.drop_prop, input=torch.tensor([0])).item() == 1
        if drop:
            g.remove((s, p, o))
g.serialize(destination=os.path.join(args.data_dir, 'corrupted_graph.nt'), format='nt')

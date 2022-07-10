import pickle
import torch
import os
from rdflib import Graph
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_fb15k237_benchmark(data):
    infile = open(os.path.join(data, 'pos_samples/answers.pickle'), 'rb')
    pos_answers = pickle.load(infile)
    infile.close()
    infile = open(os.path.join(data, 'neg_samples/answers.pickle'), 'rb')
    neg_answers = pickle.load(infile)
    infile.close()
    pos_samples = []
    files = os.listdir(os.path.join(data, 'pos_samples'))
    files.sort(key=natural_keys)
    for file in files:
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".nt"):
            g = Graph(store="Oxigraph")
            g.parse(os.path.join(data, 'pos_samples/' + filename), format="nt")
            pos_samples.append(g)

    files = os.listdir(os.path.join(data, 'neg_samples'))
    files.sort(key=natural_keys)
    neg_samples = []
    for file in files:
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".nt"):
            g = Graph(store="Oxigraph")
            g.parse(os.path.join(data,'neg_samples/' + filename), format="nt")
            neg_samples.append(g)

    return pos_samples + neg_samples, [[a] for a in pos_answers + neg_answers], None, torch.cat(
        (torch.ones(len(pos_answers)), torch.zeros(len(neg_answers))), dim=0).unsqueeze(dim=1), [[True]] * len(
        pos_answers + neg_answers), None

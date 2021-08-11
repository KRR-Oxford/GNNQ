import re
import torch
import numpy as np
from rdflib import Graph, URIRef
import argparse
import pickle

def load_triples(file):
    triples = []
    with open(file) as f:
        file_data = [re.search('(.+?)\s+(.+?)\s+(.+)\s*\.',line) for line in f.read().split('\n')[:-1]]
    # Save the triplets corresponding to only the known relations
    for line in file_data:
        triples.append([line.group(1).strip('""<> '), line.group(2).strip('""<> '), line.group(3).strip('""<> ')])
    return triples

def load_answers(path_to_answers):
    with open(path_to_answers, 'rb') as f:
        answers = pickle.load(f)
    return answers

def save_query_answers(path_to_graph, query_string, path_to_output):
    g = Graph()
    g.parse(path_to_graph, format="turtle")

    qres = g.query(query_string)

    answers = []
    for row in qres:
        answers.append([str(entity).strip() for entity in row])

    print(len(answers))

    with open(path_to_output, 'wb') as f:
        pickle.dump(answers, f)

def create_triples_with_ids(triples, relation2id=None):
    entity2id = {}

    no_rel_ids = False
    if relation2id is None:
        relation2id = {}
        no_rel_ids = True

    ent = 0
    rel = 0

    id_triples = []
    for triple in triples:
        if triple[0] not in entity2id:
            entity2id[triple[0]] = ent
            ent += 1
        if triple[2] not in entity2id:
            entity2id[triple[2]] = ent
            ent += 1
        if no_rel_ids & (triple[1] not in relation2id):
            relation2id[triple[1]] = rel
            rel += 1

        # Save the triplets corresponding to only the known relations
        if triple[1] in relation2id:
            id_triples.append([entity2id[triple[0]], relation2id[triple[1]], entity2id[triple[2]]])

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    return id_triples, entity2id, relation2id, id2entity, id2relation

def create_index_matrices(triples_with_ids):
    edges = torch.tensor(triples_with_ids).t()[[0, 2]]
    edges = torch.cat((edges, edges[[1,0]]), dim=1)
    edge_type = torch.tensor(triples_with_ids).t()[1]
    edge_type = torch.cat((edge_type, edge_type + torch.max(edge_type) + 1),dim=0)
    index_matrices_by_shape = {1: edges}
    edge_type_by_shape = {1: edge_type}
    num_edge_types_by_shape = {1: len(torch.unique(edge_type))}
    return index_matrices_by_shape, edge_type_by_shape, num_edge_types_by_shape

def add_tuples_to_index_matrices(tuples, index_matrices_by_shape , edge_type_by_shape, num_edge_types_by_shape):
    for tuple in tuples:
        tuple = torch.tensor(tuple)
        if len(tuple) - 1 not in index_matrices_by_shape:
            index_matrices_by_shape[len(tuple) - 1] = torch.stack((tuple[1:], tuple[0].repeat(len(tuple) - 1)),
                                                                   dim=0)
            edge_type_by_shape[len(tuple) - 1] = torch.tensor([0])
        else:
            index_matrices_by_shape[len(tuple) - 1] = torch.cat((index_matrices_by_shape[len(tuple) - 1], torch.stack(
                (tuple[1:], tuple[0].repeat(len(tuple) - 1)), dim=0)), dim=1)
            edge_type_by_shape[len(tuple) - 1] = torch.cat((edge_type_by_shape[len(tuple) - 1], torch.tensor([torch.max(edge_type_by_shape[len(tuple) - 1])])),
                                                            dim=0)
    for shape in edge_type_by_shape:
        if shape != 1:
            num_edge_types_by_shape[shape] = len(torch.unique(edge_type_by_shape[shape]))
    return index_matrices_by_shape, edge_type_by_shape, num_edge_types_by_shape

def create_y_vector(answers, num_nodes):
    # K - hot vector indicating all answers
    y = torch.scatter(torch.zeros(num_nodes, dtype=torch.int16), 0, torch.tensor(answers),
                      torch.ones(num_nodes, dtype=torch.int16))
    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--val_data', type=str, default='val')
    parser.add_argument('--query',type=str,default='SELECT ?s ?r ?o WHERE { ?s ?r ?o }')
    args = parser.parse_args()


    query = 'SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
    subquery = 'SELECT distinct ?v0 ?v1 ?v3 ?v4 ?v5 ?v6 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 }'
    subquery2 = 'SELECT distinct ?v4 ?v5 ?v6 ?v7 ?v8  WHERE {  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
    simple_query = 'SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4  }'
    save_query_answers(args.val_data + '/graph.ttl' , simple_query, 'val_simple_answers.pickle')
    # answers = load_answers('subquery_answers.pickle')
    print('Done')














import re
import torch
import numpy as np
from rdflib import Graph, URIRef
import argparse
import pickle
import uuid
from rdflib.plugins.sparql import prepareQuery
import anytree

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

def load_triples(file):
    triples = []
    with open(file) as f:
        file_data = [re.search('(.+?)\s+(.+?)\s+(.+)\s*\.',line) for line in f.read().split('\n')[:-1]]
    # Save the triplets corresponding to only the known relations
    for line in file_data:
        if line:
            triples.append([line.group(1).strip('""<> '), line.group(2).strip('""<> '), line.group(3).strip('""<> ')])
    return triples

def corrupt_graph(relations, path_to_graph, path_to_corrupted_graph, max_paths_length, drop_prop):
    i = 0
    g = Graph()
    g.parse(path_to_graph, format="turtle")
    for r in relations:
        for s, p, o in g:
            if str(p) == r:
                if max_paths_length[i] == 1:
                    g.add((s, URIRef(str(p) + str(1)), o))
                else:
                    new_object = URIRef("http://dummyentities.com/" + str(uuid.uuid4()))
                    g.add((s, URIRef(str(p) + str(1)), new_object))
                    new_subject = new_object
                    for j in range(2, max_paths_length[i]):
                        new_object = URIRef("http://dummyentities.com/" + str(uuid.uuid4()))
                        g.add((new_subject, URIRef(str(p) + str(j)), new_object))
                        new_subject = new_object
                    g.add((new_subject, URIRef(str(p) + str(max_paths_length[i])), o))
                drop = torch.bernoulli(p=drop_prop, input=torch.tensor([0])).item() == 1
                if drop:
                    g.remove((s,p,o))
        i = i + 1
    g.serialize(destination=path_to_corrupted_graph,format='nt')


def load_answers(path_to_answers):
    with open(path_to_answers, 'rb') as f:
        answers = pickle.load(f)
    return answers

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
    # +1 required because edge ids start with 0
    edge_type = torch.cat((edge_type, edge_type + torch.max(edge_type)  + 1),dim=0)
    index_matrices_by_shape = {1: edges}
    edge_type_by_shape = {1: edge_type}
    num_edge_types_by_shape = {1: len(torch.unique(edge_type))}
    return index_matrices_by_shape, edge_type_by_shape, num_edge_types_by_shape

def add_tuples_to_index_matrices(tuples, index_matrices_by_shape , edge_type_by_shape, num_edge_types_by_shape):
    if len(tuples[0])-1 not in index_matrices_by_shape:
        id = 0
    else:
        id = torch.max(edge_type_by_shape[len(tuples[0]) - 1]) + 1
    for tuple in tuples:
        tuple = torch.tensor(tuple)
        # If there exists no edge of this shape
        if len(tuple) - 1 not in index_matrices_by_shape:
            index_matrices_by_shape[len(tuple) - 1] = torch.stack((tuple[1:], tuple[0].repeat(len(tuple) - 1)),
                                                                   dim=0)
            edge_type_by_shape[len(tuple) - 1] = torch.tensor([id])
        else:
            index_matrices_by_shape[len(tuple) - 1] = torch.cat((index_matrices_by_shape[len(tuple) - 1], torch.stack(
                (tuple[1:], tuple[0].repeat(len(tuple) - 1)), dim=0)), dim=1)
            edge_type_by_shape[len(tuple) - 1] = torch.cat((edge_type_by_shape[len(tuple) - 1], torch.tensor([id])),
                                                            dim=0)
    for shape in edge_type_by_shape:
        if shape != 1:
            num_edge_types_by_shape[shape] = len(torch.unique(edge_type_by_shape[shape]))
    return index_matrices_by_shape, edge_type_by_shape, num_edge_types_by_shape

def create_y_vector(answers, num_nodes):
    # K - hot vector indicating all answers
    y = torch.scatter(torch.zeros(num_nodes, dtype=torch.float32), 0, torch.tensor(answers),
                      torch.ones(num_nodes, dtype=torch.float32))
    # help = y[answers]
    # y = scatter.scatter_add(src=torch.ones(num_nodes, dtype=torch.int16), index=torch.tensor(answers), out=torch.zeros(num_nodes, dtype=torch.float16), dim=0)
    return y

def create_data_object(path_to_graph, path_to_answers, paths_to_subquery_answers, base_dim, relation2id=None):
    triples = load_triples(path_to_graph)
    triples, entity2id, relation2id, _, _ = create_triples_with_ids(triples, relation2id)
    num_nodes = len(entity2id)
    path_to_answers = load_answers(path_to_answers)
    path_to_answers = [entity2id[entity[0]] for entity in path_to_answers]
    x = torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, base_dim - 1)), dim=1)
    y = create_y_vector(path_to_answers, num_nodes)
    hyperedge_indices, hyperedge_types, num_edge_types_by_shape = create_index_matrices(triples)
    for file in paths_to_subquery_answers:
        subquery_answers = load_answers(file)
        subquery_answers = [[entity2id[entity] for entity in answer] for answer in subquery_answers]
        hyperedge_indices, hyperedge_types, num_edge_types_by_shape = add_tuples_to_index_matrices(
            subquery_answers, hyperedge_indices, hyperedge_types, num_edge_types_by_shape)
    return {'hyperedge_indices':hyperedge_indices, 'hyperedge_types':hyperedge_types, 'num_edge_types_by_shape':num_edge_types_by_shape,'x':x,'y':y}, relation2id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--val_data', type=str, default='val')
    parser.add_argument('--query',type=str,default='SELECT ?s ?r ?o WHERE { ?s ?r ?o }')
    args = parser.parse_args()

    query_string = 'SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
    subquery1 = 'SELECT distinct ?v6 ?v7 ?v8  WHERE {  ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
    subquery2 = 'SELECT distinct ?v0 ?v4 ?v5 ?v6 WHERE { ?v0 <http://purl.org/stuff/rev#hasReview> ?v4 . ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4 <http://purl.org/stuff/rev#reviewer> ?v6 }'


    subqueries = [subquery1, subquery2]

    directory = 'wsdbm-data-model-2/dataset2/'
    query = 1
    save_query_answers(directory + 'graph.ttl' , query_string, directory + 'query{}/answers.pickle'.format(query))
    corrupt_graph(['http://schema.org/caption', 'http://schema.org/text', 'http://schema.org/contentRating','http://purl.org/stuff/rev#hasReview', 'http://purl.org/stuff/rev#title', 'http://purl.org/stuff/rev#reviewer', 'http://schema.org/actor', 'http://schema.org/language'], directory + "graph.ttl", directory + "corrupted_graph.ttl", [1, 2, 1, 2, 1, 1, 2, 2], 0)
    i = 0
    for subquery in subqueries:
        save_query_answers(directory + 'corrupted_graph.ttl', subquery, directory + 'query{}/subquery_answers{}.pickle'.format(query,i))
        i = i + 1
    print('Done')















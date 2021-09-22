import re
import torch
from subquery_generation import create_tree, create_subquery_trees, create_subqueries, create_all_connceted_trees
from rdflib import Graph, URIRef

def load_triples(file):
    triples = []
    with open(file) as f:
        file_data = [re.search('(.+?)\s+(.+?)\s+(.+)\s*\.',line) for line in f.read().split('\n')[:-1]]
    # Save the triplets corresponding to only the known relations
    for line in file_data:
        if line:
            triples.append([line.group(1).strip('""<> '), line.group(2).strip('""<> '), line.group(3).strip('""<> ')])
    return triples


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


def compute_query_answers(path_to_graph, query_string):
    g = Graph()
    g.parse(path_to_graph, format="nt")
    qres = g.query(query_string)
    answers = []
    for row in qres:
        answers.append([str(entity).strip() for entity in row])
    return answers

def compute_subquery_answers(path_to_corrupted_graph, query_string, subquery_depth):
    g = Graph()
    g.parse(path_to_corrupted_graph, format="nt")
    root = create_tree(query_string)
    # trees = create_subquery_trees(root, subquery_depth)
    trees = create_all_connceted_trees(root)
    subqueries = create_subqueries(trees)
    subquery_answers = []
    for subquery in subqueries:
        qres = g.query(subquery)
        answers = []
        for row in qres:
            answers.append([str(entity).strip() for entity in row])
        subquery_answers.append(answers)
    return subquery_answers

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

def create_data_object(path_to_graph, path_to_corrupted_graph, query_string, base_dim, aug, subquery_depth, relation2id=None):
    triples = load_triples(path_to_corrupted_graph)
    triples, entity2id, relation2id, _, _ = create_triples_with_ids(triples, relation2id)
    num_nodes = len(entity2id)
    x = torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, base_dim - 1)), dim=1)
    answers = compute_query_answers(path_to_graph, query_string)
    answers = [entity2id[entity[0]] for entity in answers]
    y = create_y_vector(answers, num_nodes)
    hyperedge_indices, hyperedge_types, num_edge_types_by_shape = create_index_matrices(triples)
    if aug:
        subquery_answers = compute_subquery_answers(path_to_corrupted_graph, query_string, subquery_depth)
        for answer_set in subquery_answers:
            subquery_answers = [[entity2id[entity] for entity in answer] for answer in answer_set]
            hyperedge_indices, hyperedge_types, num_edge_types_by_shape = add_tuples_to_index_matrices(
                subquery_answers, hyperedge_indices, hyperedge_types, num_edge_types_by_shape)
    return {'hyperedge_indices':hyperedge_indices, 'hyperedge_types':hyperedge_types, 'num_edge_types_by_shape':num_edge_types_by_shape,'x':x,'y':y}, relation2id















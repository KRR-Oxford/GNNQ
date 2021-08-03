import torch
import numpy as np
from rdflib import Graph, URIRef
import argparse

def data(file):
    entity2id = {}
    relation2id = {}

    ent = 0
    rel = 0

    triples = []
    with open(file) as f:
        file_data = [line.split() for line in f.read().split('\n')[:-1]]

    for triple in file_data:
        if triple[0] not in entity2id:
            entity2id[triple[0]] = ent
            ent += 1
        if triple[2] not in entity2id:
            entity2id[triple[2]] = ent
            ent += 1
        if triple[1] not in relation2id:
            relation2id[triple[1]] = rel
            rel += 1

        # Save the triplets corresponding to only the known relations
        if triple[1] in relation2id:
            triples.append([entity2id[triple[0]], relation2id[triple[1]], entity2id[triple[2]]])

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    return triples, entity2id, relation2id, id2entity, id2relation

def create_index_matrices(triples, num_rels, subquery_answers):
    edges = torch.tensor(triples).t()[[0,2]]
    edges = torch.cat((edges, edges[[1,0]]), dim=1)
    edge_type = torch.tensor(triples).t()[1]
    edge_type = torch.cat((edge_type, edge_type + torch.max(edge_type) + 1),dim=0)
    hyperedge_index = {1: edges}
    hyperedge_type = {1: edge_type}
    num_edge_types_by_shape = {1: num_rels * 2}
    for hyperedge_indices in subquery_answers:
        if len(hyperedge_indices) - 2 not in hyperedge_index:
            hyperedge_index[len(hyperedge_indices) - 2] = torch.stack((hyperedge_indices[2:], hyperedge_indices[0].repeat(len(hyperedge_indices) - 2)), dim=0)
            hyperedge_type[len(hyperedge_indices) - 2] = torch.tensor([hyperedge_indices[1]])
        else:
            hyperedge_index[len(hyperedge_indices) - 2] = torch.cat((hyperedge_index[len(hyperedge_indices) - 2], torch.stack((hyperedge_indices[2:],hyperedge_indices[0].repeat(len(hyperedge_indices)-2)),dim=0)),dim=1)
            hyperedge_type[len(hyperedge_indices) - 2] = torch.cat((hyperedge_type[len(hyperedge_indices) - 2],torch.tensor([hyperedge_indices[1]])),dim=0)
    for shape in hyperedge_type:
        if shape != 1:
            num_edge_types_by_shape[shape] = len(torch.unique(hyperedge_type[shape]))
    return hyperedge_index, hyperedge_type, num_edge_types_by_shape

def load_answers(answers_path):
    with open(answers_path, 'rb') as f:
        answers = torch.tensor(np.load(f))
    return answers

def load_subquery_answers(subquery_answer_path):
    with open(subquery_answer_path, 'rb') as f:
        subquery_answers = torch.tensor(np.load(f))
    return subquery_answers

def create_y_vector(answers, num_nodes):
    # K - hot vector indicating all answers
    y = torch.scatter(torch.zeros(num_nodes, dtype=torch.int16), 0, answers,
                      torch.ones(num_nodes, dtype=torch.int16))
    return y

def save_query_answers(path_to_graph, query_string, path_to_output):
    triples, entity2id, relation2id, id2entity, id2relation = data(path_to_graph)

    g = Graph()
    # Create an RDF URI node to use as the subject for multiple triples
    for triple in triples:
        g.add((URIRef(str(triple[0])), URIRef(str(triple[1])), URIRef(str(triple[2]))))

    qres = g.query(query_string)

    answers = []
    for row in qres:
        answers.append(int(str(row[0])))

    print(len(answers))

    with open(path_to_output, 'wb') as f:
        np.save(f, np.array(answers))


def save_subquery_answers(path_to_graph, query_string, shape, rel_type, path_to_output):
    triples, entity2id, relation2id, id2entity, id2relation = data(path_to_graph)

    g = Graph()
    # Create an RDF URI node to use as the subject for multiple triples
    for triple in triples:
        g.add((URIRef(str(triple[0])), URIRef(str(triple[1])), URIRef(str(triple[2]))))

    qres = g.query(query_string)

    answers = []
    for row in qres:
        answer = []
        for i in range(shape + 1):
            if i == 1:
                answer.append(rel_type)
            answer.append(int(str(row[i])))
        answers.append(answer)

    print(len(answers))

    with open(path_to_output, 'wb') as f:
        np.save(f, np.array(answers))

    print('Saved answers')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bla bla')
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--val_data', type=str, default='val')
    parser.add_argument('--query',type=str,default='SELECT ?s ?r ?o WHERE { ?s ?r ?o }')
    args = parser.parse_args()

    query = """SELECT distinct ?v0
                    WHERE {
                        ?v0 <28> ?v1 .
                        ?v0 <30> ?v2 .
                        ?v0 <32> ?v3 .
                        ?v0 <53> ?v4 .
                        ?v4 <9> ?v5  .
                        ?v4 <84> ?v6 .
                        ?v7 <78> ?v6 .
                        ?v7 <40> ?v8
                    }"""

    query2 = """SELECT distinct ?v0 ?v1 ?v3 ?v4 ?v5 ?v6
                    WHERE {
                        ?v0 <28> ?v1 .
                        ?v0 <32> ?v3 .
                        ?v0 <53> ?v4 .
                        ?v4 <9> ?v5  .
                        ?v4 <84> ?v6
                    }"""

    # save_subquery_answers(args.val_data + '/graph.ttl', query2, 5,0,'subquery_answers.npy')
    save_query_answers(args.val_data + '/graph.ttl', query,'answers.npy')
    print('Done')













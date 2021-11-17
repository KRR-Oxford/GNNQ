import os
import torch
from collections import defaultdict
from subquery_generation import create_tree, create_subquery_trees, create_subqueries, create_all_connceted_trees
from rdflib import Graph


def create_indices_dict(graph, entity2id=None):
    if entity2id is None:
        entity2id = {}
        ent = 0
    else:
        ent = len(entity2id)

    indices_dict = defaultdict(list)

    # Corrupted graph contains additional constants! We have to add them to the ID dictionary!
    for s, p, o in graph:
        sub = str(s).strip()
        obj = str(o).strip()
        if (sub not in entity2id):
            entity2id[sub] = ent
            ent += 1
        if (obj not in entity2id):
            entity2id[obj] = ent
            ent += 1
        indices_dict[str(p).replace('.', '')].append([entity2id[sub], entity2id[obj]])

    # indices_dict = dict(indices_dict)
    indices_dict = {**{k: torch.tensor(v).t() for k, v in indices_dict.items()},
                    **{k + "_inv": torch.tensor(v).t()[[1, 0]] for k, v in indices_dict.items()}}

    id2entity = {v: k for k, v in entity2id.items()}
    return indices_dict, entity2id, id2entity


def compute_query_answers(graph, query_string, entity2id):
    qres = graph.query(query_string)
    answers = []
    for row in qres:
        answers.append([entity2id[str(entity).strip()] for entity in row])
    return answers


def compute_subquery_answers(graph, query_string, subquery_gen_strategy, subquery_depth,
                             max_num_subquery_vars, entity2id):
    print('Generating subqueries!')
    root = create_tree(query_string)
    if subquery_gen_strategy == 'greedy':
        trees = create_subquery_trees(root, subquery_depth)
    else:
        trees = create_all_connceted_trees(root, max_num_subquery_vars)
    subqueries = create_subqueries(trees)
    subquery_answers = {}
    subquery_shape = {}
    counter = 1
    for subquery in subqueries:
        qres = graph.query(subquery)
        answers = []
        for row in qres:
            answers.append([entity2id[str(entity).strip()] for entity in row])
        answers = torch.tensor(answers)
        shape = answers.size()[1] - 1
        key = str([str(var) for var in subquery.algebra['PV']])
        subquery_answers[key] = torch.stack(
            (answers[:, 1:].flatten(), answers[:, 0].unsqueeze(1).repeat((1, shape)).flatten()), dim=0)
        subquery_shape[key] = shape
        print('Subquery {0} has {1} answers. ({2}/{3}) subqueries answered!'.format(counter, len(answers), counter,
                                                                                    len(subqueries)))
        counter = counter + 1
    return subquery_answers, subquery_shape


def create_y_vector(answers, num_nodes):
    # K - hot vector indicating all answers
    y = torch.scatter(torch.zeros(num_nodes, dtype=torch.float32), 0, torch.tensor(answers),
                      torch.ones(num_nodes, dtype=torch.float32))
    # help = y[answers]
    # y = scatter.scatter_add(src=torch.ones(num_nodes, dtype=torch.int16), index=torch.tensor(answers), out=torch.zeros(num_nodes, dtype=torch.float16), dim=0)
    return y


def create_data_object(path_to_graph, path_to_corrupted_graph, query_string, aug, subquery_gen_strategy, subquery_depth,
                       max_num_subquery_vars):
    g = Graph()
    g.parse(path_to_graph, format="nt")
    corrupted_g = Graph()
    corrupted_g.parse(path_to_corrupted_graph, format="nt")
    _, entity2id, _ = create_indices_dict(g)
    indices_dict, entity2id, _ = create_indices_dict(corrupted_g, entity2id=entity2id)
    shapes_dict = {k: 1 for k, v in indices_dict.items()}
    num_nodes = len(entity2id)
    # dummy feature vector dimension
    feat_dim = 1
    x = torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, feat_dim - 1)), dim=1)
    answers = compute_query_answers(g, query_string, entity2id)
    print(path_to_graph + ' contains {} answers for the specified query.'.format(len(answers)))
    answers = [entity[0] for entity in answers]
    y = create_y_vector(answers, num_nodes)
    observed_y = torch.zeros(num_nodes)
    observed_answers = compute_query_answers(corrupted_g, query_string, entity2id)
    print(
        path_to_corrupted_graph + ' contains {} observed answers and {} unobserved answers for the specified query.'.format(
            len(observed_answers), len(answers) - len(observed_answers)))
    if observed_answers:
        observed_answers = [entity[0] for entity in observed_answers]
        observed_y = create_y_vector(observed_answers, num_nodes)
    mask_observed = (observed_y == 0)
    if aug:
        hyper_indices_dict, hyper_shapes = compute_subquery_answers(graph=corrupted_g,
                                                                    query_string=query_string,
                                                                    subquery_gen_strategy=subquery_gen_strategy,
                                                                    subquery_depth=subquery_depth,
                                                                    max_num_subquery_vars=max_num_subquery_vars,
                                                                    entity2id=entity2id)
        indices_dict = {**indices_dict, **hyper_indices_dict}
        shapes_dict = {**shapes_dict, **hyper_shapes}
    return {'indices_dict': indices_dict, 'shapes_dict': shapes_dict, 'x': x, 'y': y,
            'mask_observed': mask_observed}


# Hyperparameters used in this function can not be tuned with optuna
def prep_data(data_directories, query_string, aug, subquery_gen_strategy, subquery_depth, max_num_subquery_vars):
    data = []
    for directory in data_directories:
        print('Preparing dataset: ' + directory)
        data_object = create_data_object(path_to_graph=os.path.join(directory, 'graph.nt'),
                                         path_to_corrupted_graph=os.path.join(directory,
                                                                              'corrupted_graph.nt'),
                                         query_string=query_string, aug=aug,
                                         subquery_gen_strategy=subquery_gen_strategy,
                                         subquery_depth=subquery_depth,
                                         max_num_subquery_vars=max_num_subquery_vars)
        data.append(data_object)
    return data

import re
import torch
from collections import defaultdict
from subquery_generation import create_tree, create_subquery_trees, create_subqueries, create_all_connceted_trees

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

    indices_dict = {**{k: torch.tensor(v).t() for k, v in indices_dict.items()},
                    **{k + "_inv": torch.tensor(v).t()[[1, 0]] for k, v in indices_dict.items()}}

    id2entity = {v: k for k, v in entity2id.items()}
    return indices_dict, entity2id, id2entity

def generate_subqueries(query_string, subquery_gen_strategy, subquery_depth,
                             max_num_subquery_vars):
    root = create_tree(query_string)
    if subquery_gen_strategy == 'greedy':
        trees = create_subquery_trees(root, subquery_depth)
    else:
        trees = create_all_connceted_trees(root, max_num_subquery_vars)
    subqueries = []
    subquery_shape = {}
    for subquery in create_subqueries(trees):
        subqueries.append(subquery)
        subquery = subquery.replace(".", "")
        shape = len(re.search("SELECT (.*) WHERE", subquery)[1].split()) - 1
        subquery_shape[subquery] = shape
    return subqueries, subquery_shape

def compute_subquery_answers(graph, entity2id, subqueries):
    subquery_answers = {}
    counter = 1
    for subquery in subqueries:
        qres = graph.query(subquery)
        answers = []
        for row in qres:
            answers.append([entity2id[str(entity).strip()] for entity in row])
        # print('Subquery {0} has {1} answers. ({2}/{3}) subqueries answered!'.format(counter, len(answers), counter,
        #                                                                             len(subqueries)))
        counter = counter + 1
        subquery = subquery.replace(".", "")
        if not answers:
            subquery_answers[subquery] = torch.tensor([])
            continue
        answers = torch.tensor(answers)
        shape = answers.size()[1] - 1
        subquery_answers[subquery] = torch.stack(
            (answers[:, 1:].flatten(), answers[:, 0].unsqueeze(1).repeat((1, shape)).flatten()), dim=0)
    return subquery_answers

def create_y_vector(answers, num_nodes):
    # k-hot vector indicating all answers
    y = torch.scatter(torch.zeros(num_nodes, dtype=torch.float32), 0, torch.tensor(answers),
                      torch.ones(num_nodes, dtype=torch.float32))
    return y

def create_data_object(labels, graph, answers, aug, subqueries):
    indices_dict, entity2id, _ = create_indices_dict(graph)
    num_nodes = len(entity2id)
    # dummy feature vector dimension
    feat_dim = 1
    try:
        x = torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, feat_dim - 1)), dim=1)
        answers = [entity2id[str(answer)] for answer in answers]
        if aug:
            hyper_indices_dict = compute_subquery_answers(graph=graph, entity2id=entity2id, subqueries=subqueries)
            indices_dict = {**indices_dict, **hyper_indices_dict}
        return {'indices_dict': indices_dict, 'x': x, 'answers': torch.tensor(answers), 'labels': torch.tensor(labels,dtype=torch.float)}
    except KeyError:
        print('Answer not in sample!')
        return None


def prep_data(pos, graphs, answers, aug, subqueries=None):
    data = []
    c = 0
    for graph in graphs:
        data_object = create_data_object([pos], graph, [answers[c]], aug=aug, subqueries=subqueries)
        if data_object:
            data.append(data_object)
            print('Loaded {0}/{1} samples!'.format(c, len(graphs)))
        c += 1
    return data

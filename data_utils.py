import re
import torch
from collections import defaultdict
from subquery_generation import create_tree, create_subqueries, create_all_connceted_trees


def create_entity2id_dict(graph, entity2id=None):
    if entity2id is None:
        entity2id = {}
        ent = 0
    else:
        ent = len(entity2id)

    # Create dictionary that maps entities to a unique id
    for s, p, o in graph:
        sub = str(s).strip()
        obj = str(o).strip()
        if (sub not in entity2id):
            entity2id[sub] = ent
            ent += 1
        if not str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            if (obj not in entity2id):
                entity2id[obj] = ent
                ent += 1

    # Create dictionary that maps ids back to entities
    id2entity = {v: k for k, v in entity2id.items()}
    return entity2id, id2entity

# Creates dictionary with edge indices for a graph
def create_indices_dict(graph, entity2id):
    indices_dict = defaultdict(list)
    for s, p, o in graph:
        if not str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            sub = str(s).strip()
            obj = str(o).strip()
            # Add edge to indices dict
            indices_dict[str(p).replace('.', '')].append([entity2id[sub], entity2id[obj]])

    # Add inverse edges for every edge
    indices_dict = {**{k: torch.tensor(v).t() for k, v in indices_dict.items()},
                    **{k + "_inv": torch.tensor(v).t()[[1, 0]] for k, v in indices_dict.items()}}
    return indices_dict


# Creates subqueries for a given query string and generation strategy
def generate_subqueries(query_string, max_num_subquery_vars):
    root = create_tree(query_string)
    trees = create_all_connceted_trees(root, max_num_subquery_vars)
    subqueries = []
    subquery_shape = {}
    for subquery in create_subqueries(trees):
        subqueries.append(subquery)
        subquery = subquery.replace(".", "")
        shape = len(re.search("SELECT (.*) WHERE", subquery)[1].split()) - 1
        subquery_shape[subquery] = shape
    return subqueries, subquery_shape


# Computes answers for a list of subqueries
def compute_subquery_answers(graph, entity2id, subqueries):
    subquery_answers = {}
    counter = 1
    for subquery in subqueries:
        qres = graph.query(subquery)
        answers = []
        for row in qres:
            answers.append([entity2id[str(entity).strip()] for entity in row])
        print('Subquery {0} has {1} answers. ({2}/{3}) subqueries answered!'.format(counter, len(answers), counter,
                                                                                    len(subqueries)))
        counter += 1
        subquery = subquery.replace(".", "")
        if not answers:
            subquery_answers[subquery] = torch.tensor([])
            continue
        answers = torch.tensor(answers)
        shape = answers.size()[1] - 1
        subquery_answers[subquery] = torch.stack(
            (answers[:, 1:].flatten(), answers[:, 0].unsqueeze(1).repeat((1, shape)).flatten()), dim=0)
    return subquery_answers

def augment_graph(indices_dict, sample_graph, entity2id, subqueries):
    hyper_indices_dict = compute_subquery_answers(graph=sample_graph, entity2id=entity2id,
                                                  subqueries=subqueries)
    return {**indices_dict, **hyper_indices_dict}

def create_feature_vectors(graph, entity2id, types):
    feat = torch.zeros(len(entity2id), len(types))
    for s, p, o in graph:
        if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            sub = str(s).strip()
            obj = str(o).strip()
            # Add edge to indices dict
            feat[entity2id[sub]][types[obj]] = 1
    return feat

# Creates data objects - dictionaries containing all information required for a sample
def create_data_object(labels, sample_graph, nodes, mask, aug, subqueries, types=None, graph=None):
    entity2id = None
    # This is unfortunatly required due to the way we create the watdiv benchmarks
    if graph:
        entity2id, _ = create_entity2id_dict(graph)
    entity2id, _ = create_entity2id_dict(sample_graph, entity2id)
    indices_dict = create_indices_dict(sample_graph, entity2id)
    num_nodes = len(nodes)
    if types:
        feat = create_feature_vectors(sample_graph, entity2id, types)
    else:
        # The benchmark queries do not contain unary predicates and thus we can ignore unary predicates
        feat = torch.zeros(num_nodes, 0)
    try:
        # Append 1 as first entry of every feature vector
        feat = torch.cat((torch.ones(num_nodes, 1), feat), dim=1)
        nodes = [entity2id[str(node)] for node in nodes]
        if aug:
           indices_dict = augment_graph(indices_dict, sample_graph, entity2id, subqueries)
        return {'indices_dict': indices_dict, 'nodes': torch.tensor(nodes), 'feat': feat,
                'labels': torch.tensor(labels, dtype=torch.float), 'mask': mask}
    except KeyError:
        print('Failed to create data object!')
        return None

def prep_data(labels, sample_graphs, nodes, masks, aug, types=None, subqueries=None, graphs=None):
    data = []
    c = 0
    for sample_graph in sample_graphs:
        if graphs:
            data_object = create_data_object(labels[c], sample_graph, nodes[c], masks[c], aug=aug,
                                             subqueries=subqueries, types=types, graph=graphs[c])
        else:
            data_object = create_data_object(labels[c], sample_graph, nodes[c], masks[c], aug=aug,
                                             subqueries=subqueries, types=types)
        # Data object might be None if the answer entity is not contained in the sample graph!
        c += 1
        if data_object:
            data.append(data_object)
            print('Loaded {0}/{1} samples!'.format(c, len(sample_graphs)))
    return data

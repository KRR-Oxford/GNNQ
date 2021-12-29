from rdflib import Graph, URIRef
from rdflib.plugins.sparql import prepareQuery
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph, rdflib_to_networkx_multidigraph
import networkx as nx
import random
from itertools import compress
from collections import defaultdict
import uuid
import torch
import copy
import pickle


def randomly_split_list(l):
    ones = int(len(l) / 2)
    zeros = len(l) - ones
    mask = [True] * ones + [False] * zeros
    random.shuffle(mask)
    inversed_mask = [not b00l for b00l in mask]
    pos = list(compress(l, mask))
    neg = list(compress(l, inversed_mask))
    return pos, neg, mask, inversed_mask


def extract_connected_subgraph_of_khop(g, start_nodes, k=2, current_iter=0):
    nodes = set()
    for entity in start_nodes:
        for edge in nx.bfs_edges(g, source=entity, depth_limit=1):
            if torch.bernoulli(torch.tensor([0.01])):
                nodes.add(edge[1])
    if current_iter < k:
        return start_nodes.union(extract_connected_subgraph_of_khop(g, start_nodes.union(nodes), k, current_iter + 1))
    else:
        return start_nodes.union(nodes)


def networkx_multidigraph_to_rdflib(networkx_graph):
    rdflib_graph = Graph()
    for edge in networkx_graph.edges:
        rdflib_graph.add((edge[0], edge[2], edge[1]))
    return rdflib_graph


def rules(rule_file='./rules.txt'):
    input_file = open(rule_file, "r")
    lines = input_file.readlines()
    rules_by_head_predicate = defaultdict(list)
    for line in lines:
        split = line.split('\t')
        rule = split[3]
        atoms = rule.split(' ')
        head_predicate = atoms[0].split('(')[0]
        head_predicate = "http://dummyrel.com" + head_predicate
        rules_by_head_predicate[head_predicate].append(rule)
    return dict(rules_by_head_predicate)


def ground_rule(g, triple, rules):
    head = str(triple[1])
    rule = random.choice(rules[head])
    atoms = rule.split(' ')
    new_object = URIRef("http://dummy.com" + str(uuid.uuid4()))
    g.add((triple[0], URIRef("http://dummyrel.com" + atoms[2].split('(')[0]), new_object))
    new_subject = new_object
    for atom in atoms[3:-1]:
        new_object = URIRef("http://dummy.com" + str(uuid.uuid4()))
        g.add((new_subject, URIRef("http://dummyrel.com" + atom.split('(')[0]), new_object))
        new_subject = new_object
    g.add((new_subject, URIRef("http://dummyrel.com" + atoms[-1].split('(')[0]), triple[2]))


def split_graph(g, mono, query):
    # rules_dict = rules()
    q = prepareQuery(query)
    qres = g.query(q)
    witness_graphs = defaultdict(list)
    witness_triples = Graph()
    print('BGP matches!')
    print(len(qres))
    for row in qres.bindings:
        dummy = Graph()
        for triple in q.algebra['p']['p']['triples']:
            dummy.add((row[triple[0]], triple[1], row[triple[2]]))
            witness_triples.add((row[triple[0]], triple[1], row[triple[2]]))
        witness_graphs[row[q.algebra['PV'][0]]].append(dummy)
    cleaned_g = g - witness_triples
    answers = list(witness_graphs.keys())
    print('Distict answers!')
    print(len(answers))
    train_answers, test_answers, _, _ = randomly_split_list(answers)
    g_train = Graph()
    # g_test = Graph()
    for triple in cleaned_g:
        if random.randint(0, 1):
            g_train.add(triple)
        # else:
        #     g_test.add(triple)
    print('All matches removed!')
    print(len(g_train.query(q)))
    # print(len(g_test.query(q)))
    witness_triples_train = Graph()
    pos_answers = []
    neg_answers = []
    random.shuffle(train_answers)
    for answer in train_answers:
        print(answer)
        # Reject witness that overlap with existing witnesses
        # reject = True
        counter = 0
        # while reject:
        for witness in witness_graphs[answer]:
            reject = False
            # witness = random.choice(witness_graphs[answer])
            for triple in witness:
                if triple in witness_triples_train:
                    reject = True
                    print('Witness rejected')
                    break
            # if counter > 20:
            #     print('No non-overlapping witness found for answer ' + answer)
            #     reject = True
            #     print('')
            #     break
            counter += 1
            if not reject:
                if random.randint(0, 1):
                    print('Added positive example')
                    pos_answers.append(witness)
                    for triple in witness:
                        g_train.add(triple)
                        witness_triples_train.add(triple)
                    break
                else:
                    print('Added negative example')
                    neg_answers.append(witness)
                    for triple in witness:
                        g_train.add(triple)
                        witness_triples_train.add(triple)
                    break
    print('Positive examples')
    print(len(pos_answers))
    print('Negative examples')
    print(len(neg_answers))


def create_samples_graphs(g, answers, witness_graphs, witness_triples, completion_rules):
    cleaned_g = g - witness_triples
    nx_cleaned_g = rdflib_to_networkx_graph(cleaned_g)
    postive_samples = []
    negative_samples = []
    for answer in answers[:10]:
        for i in range(1):
            witness_graph, witness_entities = random.choice(witness_graphs[answer])
            # Sample from cleaned_q + witness?
            sample_entities = extract_connected_subgraph_of_khop(nx_cleaned_g, witness_entities, 2)
            print('Sample entities')
            print(len(sample_entities))
            sample_graph = rdflib_to_networkx_multidigraph(cleaned_g).subgraph(list(sample_entities)).copy()
            print('Sample graph size')
            print(sample_graph.number_of_edges())
            sample_graph = networkx_multidigraph_to_rdflib(sample_graph)
            accept = False
            while not accept:
                witness_graph_copy = copy.deepcopy(witness_graph)
                if random.randint(0, 1):
                    # Corrupt witness such that it can be recovered with completion function -> positive sample
                    print('Positive sample!')
                    for triple in witness_graph:
                        if torch.bernoulli(torch.tensor([0.9])) or (str(triple[1]) not in completion_rules.keys()):
                            if torch.bernoulli(torch.tensor([0.5])) and (str(triple[1]) in completion_rules.keys()):
                                ground_rule(witness_graph_copy, triple, completion_rules)
                        else:
                            witness_graph_copy.remove(triple)
                            print('Corrupted sample!')
                            ground_rule(witness_graph_copy, triple, completion_rules)
                            accept = True
                    if accept:
                        print('Positive sample accepted!')
                        postive_samples.append(sample_graph + witness_graph_copy)
                    else:
                        print('Positive sample rejected!')
                else:
                    # Corrupt witness such that it can not be recovered with completion function -> negative sample
                    print('Negative sample!')
                    for triple in witness_graph:
                        if torch.bernoulli(torch.tensor([0.8])) or (str(triple[1]) not in completion_rules.keys()):
                            if torch.bernoulli(torch.tensor([0.5])) and (str(triple[1]) in completion_rules.keys()):
                                ground_rule(witness_graph_copy, triple, completion_rules)
                        else:
                            witness_graph_copy.remove(triple)
                            if torch.bernoulli(torch.tensor([0.5])):
                                ground_rule(witness_graph_copy, triple, completion_rules)
                            else:
                                print('Cannot be recovered!')
                                accept = True
                    if accept:
                        print('Negative sample accepted!')
                        negative_samples.append(sample_graph + witness_graph_copy)
                    else:
                        print('Negative sample rejected!')
    return postive_samples, negative_samples


def create_samples(g, query):
    completion_rules = rules()
    q = prepareQuery(query)
    qres = g.query(q)
    witness_graphs = defaultdict(list)
    witness_triples = Graph()
    print('BGP matches!')
    print(len(qres))
    for row in qres.bindings:
        dummy = Graph()
        sample_entities = set()
        for triple in q.algebra['p']['p']['triples']:
            dummy.add((row[triple[0]], triple[1], row[triple[2]]))
            witness_triples.add((row[triple[0]], triple[1], row[triple[2]]))
            sample_entities.add(row[triple[0]])
            sample_entities.add(row[triple[2]])
        witness_graphs[row[q.algebra['PV'][0]]].append((dummy, sample_entities))
    print('Constructed witness graphs!')
    answers = list(witness_graphs.keys())
    train_answers, test_answers, _, _ = randomly_split_list(answers)
    train_positive_samples, train_negative_samples = create_samples_graphs(g, train_answers, witness_graphs,
                                                                           witness_triples, completion_rules)
    outfile = open('training_samples.pkl', 'wb')
    pickle.dump((train_positive_samples, train_negative_samples), outfile)
    outfile.close()
    test_positive_samples, test_negative_samples = create_samples_graphs(g, test_answers, witness_graphs,
                                                                         witness_triples, completion_rules)
    outfile = open('test_samples.pkl', 'wb')
    pickle.dump((test_positive_samples, test_negative_samples), outfile)
    outfile.close()


# def create_core_graphs(g, query):
#     q = prepareQuery(query)
#     qres = g.query(q)
#     print('Answered query')
#     core_graphs = []
#     answers = []
#     print(len(qres))
#     c = 0
#     for row in qres.bindings:
#         g = Graph()
#         i = 0
#         for triple in q.algebra['p']['p']['triples']:
#             g.add((row[triple[0]],triple[1],row[triple[2]]))
#             i+=2
#         core_graphs.append(g)
#         answers.append(row[0])
#     return core_graphs, answers


if __name__ == '__main__':
    graph = 'datasets/fb15k237/dataset1/graph.nt'

    g = Graph()
    g.parse(graph, format="nt")

    # query = 'CONSTRUCT WHERE { ?film <http://dummyrel.com/film/film/genre> ?genre . ?film <http://dummyrel.com/film/film/country> ?country . ?genre <http://dummyrel.com/media_common/netflix_genre/titles> ?titles . ?country <http://dummyrel.com/location/country/official_language> ?language . ?country2 <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?country . ?country2 <http://dummyrel.com/organization/organization_member/member_of./organization/organization_membership/organization> ?org . ?event <http://dummyrel.com/base/culturalevent/event/entity_involved> ?org }'
    mono_query = 'select distinct ?film where { ?film <http://dummyrel.com/film/film/genre> ?genre . ?film <http://dummyrel.com/film/film/country> ?country . ?genre <http://dummyrel.com/media_common/netflix_genre/titles> ?titles . ?country <http://dummyrel.com/location/country/official_language> ?language . ?country2 <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?country . ?country2 <http://dummyrel.com/organization/organization_member/member_of./organization/organization_membership/organization> ?org . ?event <http://dummyrel.com/base/culturalevent/event/entity_involved> ?org }'
    mono_query = 'select distinct ?org where { ?org <http://dummyrel.com/organization/organization/headquarters./location/mailing_address/state_province_region> ?region . ?biggerregion <http://dummyrel.com/location/location/contains> ?region . ?biggerregion <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?neighbourregion . ?biggerregion <http://dummyrel.com/location/country/capital> ?capital . ?neighbourregion <http://dummyrel.com/location/country/official_language> ?lang . ?capital <http://dummyrel.com/common/topic/webpage./common/webpage/category> ?category . ?capital <http://dummyrel.com/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month> ?month }'
    bgp_query = 'select ?film ?genre ?film ?country ?genre ?titles ?country ?language ?country2 ?country ?country2 ?org ?event ?org where { ?film <http://dummyrel.com/film/film/genre> ?genre . ?film <http://dummyrel.com/film/film/country> ?country . ?genre <http://dummyrel.com/media_common/netflix_genre/titles> ?titles . ?country <http://dummyrel.com/location/country/official_language> ?language . ?country2 <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?country . ?country2 <http://dummyrel.com/organization/organization_member/member_of./organization/organization_membership/organization> ?org . ?event <http://dummyrel.com/base/culturalevent/event/entity_involved> ?org }'
    bgp_query = 'select ?org ?region ?biggerregion ?region ?biggerregion ?neighbourregion ?biggerregion ?capital ?neighbourregion ?lang ?capital ?category ?capital ?month where { ?org <http://dummyrel.com/organization/organization/headquarters./location/mailing_address/state_province_region> ?region . ?biggerregion <http://dummyrel.com/location/location/contains> ?region . ?biggerregion <http://dummyrel.com/location/location/adjoin_s./location/adjoining_relationship/adjoins> ?neighbourregion . ?biggerregion <http://dummyrel.com/location/country/capital> ?capital . ?neighbourregion <http://dummyrel.com/location/country/official_language> ?lang . ?capital <http://dummyrel.com/common/topic/webpage./common/webpage/category> ?category . ?capital <http://dummyrel.com/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month> ?month }'

    # graphs = split_graph(g, mono_query ,bgp_query)
    create_samples(g, bgp_query)
    # compute_k_hop_neighbourhood(core_graphs)
    print('Done')

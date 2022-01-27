import torch
import os
import pickle
import uuid
import random
from rdflib import Graph, URIRef
from argparse import ArgumentParser
from collections import defaultdict

def generate_rules(head_predicates, num_rules=3, max_length=2):
    rules_dict = defaultdict(list)
    for predicate in head_predicates:
        for i in range(num_rules):
            body_predicate = predicate + "_" +str(uuid.uuid4())
            l = torch.randint(low=1, high=max_length + 1, size=(1,)).item()
            rule_body = []
            for j in range(1,l + 1):
                rule_body.append(body_predicate + "_{}".format(j))
            rules_dict[predicate].append(rule_body)
    with open('../datasets/watdiv/rules_dict.pickle', 'wb') as f:
        pickle.dump(dict(rules_dict), f)

# Todo: Change function to randomly delete edges from the graph
def corrupt_graph(data_directory, drop_prop):
    g = Graph()
    g.parse(os.path.join(data_directory, 'graph.nt'), format="nt")
    with open('../datasets/watdiv/rules_dict.pickle', 'rb') as f:
        rules_dict = pickle.load(f)
    for s, p, o in g:
        if str(p) in rules_dict.keys():
            rule_body = random.choice(rules_dict[str(p)])
            grounded_predicates = Graph()
            if len(rule_body) == 1:
                grounded_predicates.add((s, URIRef(rule_body[0]), o))
            else:
                new_object = URIRef("http://dummyentities.com/" + str(uuid.uuid4()))
                grounded_predicates.add((s, URIRef(rule_body[0]), new_object))
                new_subject = new_object
                for j in range(1, len(rule_body) - 1):
                    new_object = URIRef("http://dummyentities.com/" + str(uuid.uuid4()))
                    grounded_predicates.add((new_subject, URIRef(rule_body[j]), new_object))
                    new_subject = new_object
                grounded_predicates.add((new_subject, URIRef(rule_body[len(rule_body) - 1]), o))
            drop = torch.bernoulli(torch.tensor([drop_prop])).item() == 1
            if drop:
                g.remove((s, p, o))
                for triple in grounded_predicates:
                    g.add(triple)
            else:
                if torch.bernoulli(torch.tensor([0.5])):
                    for triple in grounded_predicates:
                        g.add(triple)


    g.serialize(destination=os.path.join(data_directory, 'corrupted_graph.nt'), format='nt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', help="File with all the rules", default='../datasets/watdiv/dataset7')
    args = parser.parse_args()

    # Remember to use the same path_length_dict for all datasets
    rels = ['http://schema.org/caption', 'http://schema.org/text', 'http://schema.org/contentRating',
         'http://purl.org/stuff/rev#title', 'http://purl.org/stuff/rev#reviewer', 'http://schema.org/actor',
         'http://schema.org/language', 'http://schema.org/legalName', 'http://purl.org/goodrelations/offers',
         'http://schema.org/eligibleRegion', 'http://purl.org/goodrelations/includes', 'http://schema.org/jobTitle',
         'http://xmlns.com/foaf/homepage', 'http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase',
         'http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor', 'http://purl.org/stuff/rev#hasReview',
         'http://purl.org/stuff/rev#totalVotes']
    path_length_dict = corrupt_graph(args.data_dir, 0.2)
    # generate_rules(rels)
    print('Done')

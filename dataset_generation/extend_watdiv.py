import torch
import os
import pickle
import uuid
from rdflib import Graph, URIRef
from argparse import ArgumentParser

# Todo: Change function to randomly delete edges from the graph
def corrupt_graph(head_relations, data_directory, max_path_length, drop_prop,
                  path_length_dict_directory=None):
    save_dict = False
    g = Graph()
    g.parse(os.path.join(data_directory, 'graph.nt'), format="nt")
    if not path_length_dict_directory:
        path_length_dict = {}
        for r in head_relations:
            path_length = torch.randint(low=1, high=max_path_length + 1, size=(1,)).item()
            path_length_dict[str(r)] = path_length
        save_dict = True
    else:
        with open(os.path.join(path_length_dict_directory, 'path_length_dict.pickle'), 'rb') as f:
            path_length_dict = pickle.load(f)
        assert len(head_relations) == len(path_length_dict)
    for s, p, o in g:
        if str(p) in head_relations:
            path_length = path_length_dict[str(p)]
            if path_length == 1:
                g.add((s, URIRef(str(p) + str(1)), o))
            else:
                new_object = URIRef("http://dummyentities.com/" + str(uuid.uuid4()))
                g.add((s, URIRef(str(p) + str(1)), new_object))
                new_subject = new_object
                for j in range(2, path_length):
                    new_object = URIRef("http://dummyentities.com/" + str(uuid.uuid4()))
                    g.add((new_subject, URIRef(str(p) + str(j)), new_object))
                    new_subject = new_object
                g.add((new_subject, URIRef(str(p) + str(path_length)), o))
            drop = torch.bernoulli(p=drop_prop, input=torch.tensor([0])).item() == 1
            if drop:
                g.remove((s, p, o))
    g.serialize(destination=os.path.join(data_directory, 'corrupted_graph.nt'), format='nt')
    if save_dict:
        with open(os.path.join(data_directory, 'path_length_dict.pickle'), 'wb') as f:
            pickle.dump(path_length_dict, f)
    return path_length_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', help="File with all the rules", default='../datasets/wsdbm-data-model-v1/dataset1')
    args = parser.parse_args()

    # Remember to use the same path_length_dict for all datasets
    rels = ['http://schema.org/caption', 'http://schema.org/text', 'http://schema.org/contentRating',
         'http://purl.org/stuff/rev#title', 'http://purl.org/stuff/rev#reviewer', 'http://schema.org/actor',
         'http://schema.org/language', 'http://schema.org/legalName', 'http://purl.org/goodrelations/offers',
         'http://schema.org/eligibleRegion', 'http://purl.org/goodrelations/includes', 'http://schema.org/jobTitle',
         'http://xmlns.com/foaf/homepage', 'http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase',
         'http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor', 'http://purl.org/stuff/rev#hasReview',
         'http://purl.org/stuff/rev#totalVotes']
    path_length_dict = corrupt_graph(rels, args.data_dir, 2, 0.1)
    print('Done')

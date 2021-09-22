import torch
import pickle
import pickle
import uuid
from rdflib import Graph, URIRef

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

def load_answers(path_to_answers):
    with open(path_to_answers, 'rb') as f:
        answers = pickle.load(f)
    return answers

# Todo: Change function to randomly delete edges from the graph
def corrupt_graph(head_relations, path_to_graph, path_to_corrupted_graph, max_path_length, drop_prop, path_length_dict=None):
    g = Graph()
    g.parse(path_to_graph, format="nt")
    if not path_length_dict:
        path_length_dict = {}
    else:
        assert len(head_relations) == len(path_length_dict)
    for s, p, o in g:
        if str(p) in head_relations:
            if str(p) not in path_length_dict.keys():
                path_length = torch.randint(low=1, high=max_path_length+1, size=(1,)).item()
                path_length_dict[str(p)] = path_length
            else:
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
            g.remove((s,p,o))
    g.serialize(destination=path_to_corrupted_graph,format='nt')
    return path_length_dict

if __name__ == '__main__':
    directory = 'wsdbm-data-model-2/dataset1/'
    path_length_dict = corrupt_graph(['http://schema.org/caption', 'http://schema.org/text', 'http://schema.org/contentRating', 'http://purl.org/stuff/rev#title', 'http://purl.org/stuff/rev#reviewer','http://schema.org/actor', 'http://schema.org/language','http://schema.org/legalName','http://purl.org/goodrelations/offers','http://schema.org/eligibleRegion','http://purl.org/goodrelations/includes','http://schema.org/jobTitle','http://xmlns.com/foaf/homepage','http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase','http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor','http://purl.org/stuff/rev#hasReview','http://purl.org/stuff/rev#totalVotes'], directory + "graph.nt", directory + "corrupted_graph.nt", 3, 0.05)
    directory = 'wsdbm-data-model-2/dataset2/'
    corrupt_graph(
        ['http://schema.org/caption', 'http://schema.org/text', 'http://schema.org/contentRating', 'http://purl.org/stuff/rev#title', 'http://purl.org/stuff/rev#reviewer','http://schema.org/actor', 'http://schema.org/language','http://schema.org/legalName','http://purl.org/goodrelations/offers','http://schema.org/eligibleRegion','http://purl.org/goodrelations/includes','http://schema.org/jobTitle','http://xmlns.com/foaf/homepage','http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase','http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor','http://purl.org/stuff/rev#hasReview','http://purl.org/stuff/rev#totalVotes'], directory + "graph.nt",
        directory + "corrupted_graph.nt", 3, 0.05, path_length_dict)
    print('Done')
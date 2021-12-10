from rdflib import Graph, URIRef
import os
import re

# Create a Graph
g = Graph()

for file in ['train.txt', 'valid.txt', 'test.txt']:
    with open(os.path.join('../datasets/YAGO3-10', file)) as f:
        lines = f.readlines()

        for line in lines:
            triple = re.split(' |\t', line.replace('\\u', '').strip())
            triple = (URIRef("http://dummyurl.com/" + triple[0]), URIRef("http://dummyrel.com/" + triple[1]),
                      URIRef("http://dummyurl.com/" + triple[2]))
            # triple = (URIRef("http://dummyurlA.com" + t) for t in triple)
            g.add(triple)

current_directory = os.getcwd()
g.serialize(destination='../datasets/YAGO3-10/graph.nt', format='nt')

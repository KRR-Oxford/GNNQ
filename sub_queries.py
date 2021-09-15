from rdflib.plugins.sparql import prepareQuery
from rdflib import Graph, Variable, URIRef
from anytree import Node, NodeMixin, RenderTree, PreOrderIter
import math
import copy

class CustomNode(Node):
    def __init__(self, name, is_rel=False, is_inv=False, parent=None, children=None):
        super().__init__(name, parent, children)
        self.is_rel = is_rel
        self.is_inv = is_inv
    separator = "|"

def create_tree(query_string):
    q = prepareQuery(query_string)
    answer_variable = str(q.algebra.p.p.PV[0])
    root = CustomNode(answer_variable)
    node = root
    triples = [[str(triple[0]),str(triple[1]),str(triple[2])] for triple in q.algebra.p.p.p.triples]
    children =[]
    visited = set()
    while node != None:
        for triple in triples:
            if triple[0] == node.name and not triple[2] in visited:
                rel = CustomNode(triple[1], parent=node, is_rel=True)
                children.append(CustomNode(triple[2] , parent=rel))
            if triple[2] == node.name and not triple[0] in visited:
                rel = CustomNode(triple[1], parent=node, is_rel=True, is_inv=True)
                children.append(CustomNode(triple[0], parent=rel))
        if len(children) > 0:
            visited.add(node.name)
            node = children.pop(0)
        else:
            node = None
    return root

def max_depth(node):
    deep = node
    for leaf in node.leaves:
        if deep.depth < leaf.depth:
            deep = leaf
    return deep

def algo(root, subquery_depth):
    l = max_depth(root)
    if math.floor(l.depth/2) < 2 or subquery_depth < 2:
        return []
    if math.floor(l.depth/2) < subquery_depth:
        return [root]
    c = 1
    cp_p = None
    while c <= subquery_depth:
        rel = l.parent
        p = rel.parent
        if c == subquery_depth and p.children:
            cp_p = CustomNode(p.name)
            rel.parent = cp_p
            for child in p.children:
                if not child.children[0].is_leaf:
                    child.parent = cp_p
                else:
                    child.parent = None
        l = p
        c = c + 1
    return algo(root, subquery_depth) + [cp_p]

def create_queries(subs):
    queries = []
    for sub in subs:
        triples = []
        vars = []
        for node in PreOrderIter(sub):
            if node.is_rel and node.is_inv:
                triples.append((Variable(node.children[0].name), URIRef(node.name), Variable(node.parent.name)))
            elif node.is_rel:
                triples.append((Variable(node.parent.name),URIRef(node.name),Variable(node.children[0].name)))
            else:
                vars.append(Variable(node.name))
        q = prepareQuery('SELECT distinct ?v0 WHERE { ?v0 ?v1 ?v2}')
        q.algebra['PV'] = vars
        q.algebra['_vars'] = set(vars)
        q.algebra['p']['PV'] = vars
        q.algebra['p']['_vars'] = set(vars)
        q.algebra['p']['p']['PV'] = vars
        q.algebra['p']['p']['_vars'] = set(vars)
        q.algebra['p']['p']['p']['triples'] = triples  # list of tuples
        q.algebra['p']['p']['p']['_vars'] = set(vars)
        queries.append(q)
    return queries

g = Graph()
g.parse('./GNNQ/wsdbm-data-model-2/dataset1/graph.ttl', format="turtle")
# query ='SELECT distinct ?v4 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
query = 'SELECT distinct ?v8 WHERE { ?v0 <http://schema.org/legalName> ?v1 . ?v0 <http://purl.org/goodrelations/offers> ?v2 . ?v2  <http://schema.org/eligibleRegion> ?v10 . ?v2  <http://purl.org/goodrelations/includes> ?v3 . ?v4 <http://schema.org/jobTitle> ?v5 . ?v4 <http://xmlns.com/foaf/homepage> ?v6 . ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v7 . ?v7 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v3 . ?v3 <http://purl.org/stuff/rev#hasReview> ?v8 . ?v8 <http://purl.org/stuff/rev#totalVotes> ?v9 .}'
root = create_tree(query)
# # for pre, fill, node in RenderTree(root):
# #     print("%s%s" % (pre, node.name))
subs = algo(root,2)
for sub in subs:
    for pre, fill, node in RenderTree(sub):
        print("%s%s" % (pre, node.name))
queries = create_queries(subs)
for query in queries:
    qres = g.query(query)
    print(len(qres))



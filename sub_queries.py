from rdflib.plugins.sparql import prepareQuery
from anytree import Node, RenderTree
import math

class CustomNode(Node):
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
                rel = CustomNode(triple[1], parent=node)
                children.append(CustomNode(triple[2] , parent=rel))
            if triple[2] == node.name and not triple[0] in visited:
                rel = CustomNode(triple[1] + '_inv', parent=node)
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
        l = p
        c = c + 1
    return algo(root, subquery_depth) + [cp_p]

query ='SELECT distinct ?v8 WHERE { ?v0 <http://schema.org/legalName> ?v1 . ?v0 <http://purl.org/goodrelations/offers> ?v2 . ?v2  <http://schema.org/eligibleRegion> ?v10 . ?v2  <http://purl.org/goodrelations/includes> ?v3 . ?v4 <http://schema.org/jobTitle> ?v5 . ?v4 <http://xmlns.com/foaf/homepage> ?v6 . ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v7 . ?v7 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v3 . ?v3 <http://purl.org/stuff/rev#hasReview> ?v8 . ?v8 <http://purl.org/stuff/rev#totalVotes> ?v9 .}'
root = create_tree(query)
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
subs = algo(root,2)
print('Hello')
for sub in subs:
    for pre, fill, node in RenderTree(sub):
        print("%s%s" % (pre, node.name))


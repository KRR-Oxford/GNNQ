from rdflib.plugins.sparql import prepareQuery
from rdflib import Graph, Variable, URIRef
from anytree import Node, RenderTree, LevelOrderIter, find_by_attr
import itertools
import math


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
    triples = [[str(triple[0]), str(triple[1]), str(triple[2])] for triple in q.algebra.p.p.p.triples]
    children = []
    visited = set()
    while node != None:
        for triple in triples:
            if triple[0] == node.name and not triple[2] in visited:
                rel = CustomNode(triple[1], parent=node, is_rel=True)
                children.append(CustomNode(triple[2], parent=rel))
            if triple[2] == node.name and not triple[0] in visited:
                rel = CustomNode(triple[1], parent=node, is_rel=True, is_inv=True)
                children.append(CustomNode(triple[0], parent=rel))
        if len(children) > 0:
            visited.add(node.name)
            node = children.pop(0)
        else:
            node = None
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))
    return root


# Egor's approach

def compute_subquery_nodes_root(node):
    if not node.children:
        return [node.name]
    res = [node.name]
    for child in node.children:
        res = list(itertools.product(res, ["_"] + compute_subquery_nodes_root(child.children[0])))
    return res


def compute_subquery_nodes_no_root(node):
    if not node.children:
        return []
    res = []
    for child in node.children:
        res = res + compute_subquery_nodes_root(child.children[0]) + compute_subquery_nodes_no_root(child.children[0])
    return res


def flatten2list(object):
    gather = []
    for item in object:
        if isinstance(item, (list, tuple, set)):
            gather.extend(flatten2list(item))
        else:
            gather.append(item)
    return gather


def compute_subquery_nodes(node):
    unflattened = compute_subquery_nodes_root(node) + compute_subquery_nodes_no_root(node)
    res = []
    for i in unflattened:
        h = flatten2list([i])
        h = filter(lambda el: el != "_", h)
        res.append(list(h))
    return res


def create_subtree_from_nodeset(root, nodes):
    new_root = None
    for node in LevelOrderIter(root):
        if node.name in nodes:
            if not new_root:
                new_root = CustomNode(node.name)
            else:
                rel = node.parent
                h = CustomNode(rel.name, parent=find_by_attr(new_root, rel.parent.name), is_rel=True, is_inv=rel.is_inv)
                CustomNode(node.name, parent=h)
    return new_root


def create_all_connceted_trees(root, max_num_subquery_vars=4):
    subquery_nodes = compute_subquery_nodes(root)
    trees = []
    for node_set in subquery_nodes:
        tree = create_subtree_from_nodeset(root, node_set)
        if tree.height > 2 and len(tree.descendants) / 2 < max_num_subquery_vars:
            for pre, fill, node in RenderTree(tree):
                print("%s%s" % (pre, node.name))
            trees.append(tree)
    return trees


# My approach
def max_depth(node):
    deep = node
    for leaf in node.leaves:
        if deep.depth < leaf.depth:
            deep = leaf
    return deep


def create_subquery_trees(root, subquery_depth):
    l = max_depth(root)
    if math.floor(l.depth / 2) < 2 or subquery_depth < 2:
        return []
    if math.floor(l.depth / 2) < subquery_depth:
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
    for pre, fill, node in RenderTree(cp_p):
        print("%s%s" % (pre, node.name))
    return create_subquery_trees(root, subquery_depth) + [cp_p]


def create_subqueries(trees):
    queries = []
    for tree in trees:
        triples = []
        vars = []
        for node in LevelOrderIter(tree):
            if node.is_rel and node.is_inv:
                triples.append((Variable(node.children[0].name), URIRef(node.name), Variable(node.parent.name)))
            elif node.is_rel:
                triples.append((Variable(node.parent.name), URIRef(node.name), Variable(node.children[0].name)))
            else:
                vars.append(Variable(node.name))
        q = prepareQuery('SELECT ?s ?p ?o WHERE { ?s ?p ?o}')  # dummy query to create a query object
        q.algebra['PV'] = vars
        q.algebra['_vars'] = set(vars)
        q.algebra['p']['PV'] = vars
        q.algebra['p']['_vars'] = set(vars)
        q.algebra['p']['p']['triples'] = triples  # list of tuples
        q.algebra['p']['p']['_vars'] = set(vars)
        queries.append(q)
    return queries


if __name__ == '__main__':
    # g = Graph()
    # g.parse('./GNNQ/wsdbm-data-model-2/dummy/corrupted_graph.nt', format='nt')
    query = 'SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
    root = create_tree(query)
    # trees = create_subquery_trees(root,2)
    trees = create_all_connceted_trees(root, 4)
    print('{} sub-queries have been created!'.format(len(trees)))
    queries = create_subqueries(trees)
    # for query in queries:
    #     qres = g.query(query)
    #     print(len(qres))

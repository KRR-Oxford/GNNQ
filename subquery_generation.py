from rdflib.plugins.sparql import prepareQuery
from rdflib import Variable, URIRef
from anytree import Node, RenderTree, LevelOrderIter, find_by_attr
import itertools

# Extension of the anytree node class
class CustomNode(Node):
    def __init__(self, name, is_rel=False, is_inv=False, is_leaf_in_q=False, parent=None, children=None):
        super().__init__(name, parent, children)
        self.is_rel = is_rel
        self.is_inv = is_inv
        self.is_leaf_in_q = is_leaf_in_q

    separator = "|"

# Creates an anytree tree representing a given query string
def create_tree(query_string):
    q = prepareQuery(query_string)
    answer_variable = str(q.algebra.p.p.PV[0])
    root = CustomNode(answer_variable)
    node = root
    triples = [[str(triple[0]), str(triple[1]), str(triple[2])] for triple in q.algebra.p.p.p.triples]
    children = []
    visited = set()
    while node != None:
        leaf = True
        for triple in triples:
            if triple[0] == node.name and not triple[2] in visited:
                leaf = False
                rel = CustomNode(triple[1], parent=node, is_rel=True)
                children.append(CustomNode(triple[2], parent=rel))
            if triple[2] == node.name and not triple[0] in visited:
                leaf = False
                rel = CustomNode(triple[1], parent=node, is_rel=True, is_inv=True)
                children.append(CustomNode(triple[0], parent=rel))
        if leaf:
            node.is_leaf_in_q = leaf
        if len(children) > 0:
            visited.add(node.name)
            node = children.pop(0)
        else:
            node = None
    for pre, fill, node in RenderTree(root):
        print("%s%s%s" % (pre, node.name, "_inv" if node.is_inv else ""))
    return root


# Computes a list variables that occur together in a connected subquery which contains the answer variable
def compute_subquery_nodes_root(node):
    if not node.children:
        return [node.name]
    res = [node.name]
    for child in node.children:
        res = list(itertools.product(res, ["_"] + compute_subquery_nodes_root(child.children[0])))
    return res


# Computes all variables that occur together in a connected subquery which does not contain the answer variable
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


# Computes a list of lists where each element specifies the variables that occur together in a connected subquery
def compute_subquery_nodes(node):
    unflattened = compute_subquery_nodes_root(node) + compute_subquery_nodes_no_root(node)
    res = []
    for i in unflattened:
        h = flatten2list([i])
        h = filter(lambda el: el != "_", h)
        res.append(list(h))
    return res


# Creates anytree tree representing a subquery from a list of variables
def create_subtree_from_nodeset(root, nodes):
    new_root = None
    for node in LevelOrderIter(root):
        if node.name in nodes:
            if not new_root:
                new_root = CustomNode(node.name)
            else:
                rel = node.parent
                h = CustomNode(rel.name, parent=find_by_attr(new_root, rel.parent.name), is_rel=True, is_inv=rel.is_inv)
                CustomNode(node.name, parent=h, is_leaf_in_q=node.is_leaf_in_q)
    return new_root


# Creates a list containing an anytree tree for every connected subquery
def create_all_connceted_trees(root, max_num_subquery_vars=100):
    subquery_nodes = compute_subquery_nodes(root)
    trees = []
    for node_set in subquery_nodes:
        tree = create_subtree_from_nodeset(root, node_set)
        if tree.height > 2 and len(tree.descendants) / 2 < max_num_subquery_vars:
            for pre, fill, node in RenderTree(tree):
                print("%s%s" % (pre, node.name))
            trees.append(tree)
    return trees


# Creates a SPARQL string given an anytree tree representation of a subquery
def create_subqueries(trees, all_vars=False):
    queries = []
    for tree in trees:
        triples = []
        vars = []
        for node in LevelOrderIter(tree):
            if node.is_rel and node.is_inv:
                triples.append((Variable(node.children[0].name), URIRef(node.name), Variable(node.parent.name)))
            elif node.is_rel:
                triples.append((Variable(node.parent.name), URIRef(node.name), Variable(node.children[0].name)))
            elif all_vars or not node.parent or (not node.children and not node.is_leaf_in_q):
                vars.append(Variable(node.name))
        answer_vars = ''
        for var in vars:
            answer_vars += '?' + str(var) + ' '
        bgp = ''
        for triple in triples:
            bgp += '?' + str(triple[0]) + ' <' + str(triple[1]) + '> ' + '?' + str(triple[2]) + ' . '
        subquery_string = 'SELECT DISTINCT ' + answer_vars + ' WHERE { ' + bgp + ' }'
        queries.append(subquery_string)
    return queries


if __name__ == '__main__':
    query = 'SELECT distinct ?v2 WHERE { ?v0 <http://schema.org/legalName> ?v1 . ?v0 <http://purl.org/goodrelations/offers> ?v2 . ?v2  <http://schema.org/eligibleRegion> ?v10 . ?v2  <http://purl.org/goodrelations/includes> ?v3 . ?v4 <http://schema.org/jobTitle> ?v5 . ?v4 <http://xmlns.com/foaf/homepage> ?v6 . ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v7 . ?v7 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v3 . ?v3 <http://purl.org/stuff/rev#hasReview> ?v8 . ?v8 <http://purl.org/stuff/rev#totalVotes> ?v9 }'
    root = create_tree(query)
    trees = create_all_connceted_trees(root, 8)
    print('{} sub-queries have been created!'.format(len(trees)))
    queries = create_subqueries(trees)

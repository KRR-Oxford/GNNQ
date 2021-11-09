import unittest
import torch
from model import HGNNLayer

class ModelTestCase(unittest.TestCase):

    def test_message_passing(self):
        base_dim = 16
        x = torch.ones(base_dim)
        x = torch.diag(x)

        shapes_dict = {'1': 1, '2': 1, '3':2, '4':2, '5':3, '6':3 }
        indices_dict = {'1': torch.tensor([[0, 1, 2, 3],
                                                 [5, 5, 7, 9]], dtype=torch.long),
                                '2': torch.tensor([[4, 5],
                                                 [9, 10]], dtype=torch.long),
                                '3': torch.tensor([[0, 1, 3, 4],
                                                 [2, 2, 2, 2]], dtype=torch.long),
                                '4': torch.tensor([[6, 7],
                                                   [8, 8]], dtype=torch.long),
                                '5': torch.tensor([[9, 10, 11],
                                                 [12, 12, 12]], dtype=torch.long),
                                '6': torch.tensor([[11, 13, 14],
                                                 [15, 15, 15]], dtype=torch.long)
                                }

        layer = HGNNLayer(base_dim, base_dim, shapes_dict)
        layer.C.weight.data = torch.diag(torch.ones(base_dim))
        layer.C.bias.data = torch.zeros(base_dim)
        for edge, shape in shapes_dict.items():
            layer.A[edge].data = torch.diag(torch.ones(base_dim)).repeat(shape, 1)

        out = layer(x, indices_dict, shapes_dict)
        self.assertTrue(torch.equal(out, torch.tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 1.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000]])))

from data_utils import create_indices_dict, compute_subquery_answers
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

class QueryTestCase(unittest.TestCase):

    def test_subquery_answers(self):
         g = Graph()
         g.parse('datasets/dummy/corrupted_graph.nt', format="nt")
         _, entity2id, _ = create_indices_dict(g)
         subquery = prepareQuery('SELECT ?v0 ?v4 ?v2 ?v5 WHERE { ?v0  <http://schema.org/text> ?v2 . ?v0  <http://purl.org/stuff/rev#hasReview> ?v4 . ?v4 <http://purl.org/stuff/rev#title> ?v5 }')
         key = str([str(var) for var in subquery.algebra['PV']])
         qres = g.query(subquery)
         answers = []
         for row in qres:
             answers.append([entity2id[str(entity).strip()] for entity in row])
         answers = torch.tensor(answers)
         answers = torch.stack((answers[:, 1:].flatten(), answers[:, 0].unsqueeze(1).repeat((1, answers.size()[1] - 1)).flatten()), dim=0)
         query = 'SELECT distinct ?v0 WHERE { ?v0  <http://schema.org/caption> ?v1 . ?v0   <http://schema.org/text> ?v2 . ?v0 <http://schema.org/contentRating> ?v3 . ?v0   <http://purl.org/stuff/rev#hasReview> ?v4 .  ?v4 <http://purl.org/stuff/rev#title> ?v5 . ?v4  <http://purl.org/stuff/rev#reviewer> ?v6 . ?v7 <http://schema.org/actor> ?v6 . ?v7 <http://schema.org/language> ?v8  }'
         subquery_answers, _ = compute_subquery_answers(g, query, 'not greedy', 2, 6, entity2id)
         answers2 = subquery_answers[key]
         self.assertTrue(torch.equal(answers, answers2))



if __name__ == '__main__':
    unittest.main()

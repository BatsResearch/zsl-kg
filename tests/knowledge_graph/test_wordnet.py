import unittest
import torch

from allennlp.common.params import Params
from zsl_kg.knowledge_graph.wordnet import WordNetKG


class TestWordNetKG(unittest.TestCase):
    def setUp(
        self,
    ):
        """creates an instance of KG with sample data."""
        params = Params({})
        nodes = [
            "/c/en/cat",
            "/c/en/dog",
            "/c/en/elephant",
        ]
        # (u, v)
        edges = [
            (
                0,
                1,
            ),
            (
                0,
                2,
            ),
            (
                1,
                2,
            ),
        ]
        features = torch.randn((3, 10))
        self.kg_obj = WordNetKG(
            nodes,
            features,
            edges,
            params,
        )

        self.bidirectional_kg_obj = WordNetKG(
            nodes,
            features,
            edges,
            Params({"bidirectional": True}),
        )

    def test_features(
        self,
    ):

        # check first for non empty; check the dimensions
        self.assertEqual(
            self.kg_obj.features.size(0),
            3,
        )
        self.assertEqual(
            self.kg_obj.features.size(1),
            10,
        )

    def test_adj_lists(self):
        # test graph
        self.kg_obj.setup_graph()

        self.assertEqual(self.kg_obj.adj_lists[0], set([1, 2]))
        self.assertEqual(self.kg_obj.adj_lists[1], set([2]))

    def test_bidrectional_adj_lists(self):
        # test graph
        self.kg_obj.setup_graph(bidirectional=True)

        self.assertEqual(set(self.kg_obj.adj_lists[0]), set([1, 2]))
        self.assertEqual(set(self.kg_obj.adj_lists[1]), set([2, 0]))
        self.assertEqual(set(self.kg_obj.adj_lists[2]), set([1, 0]))

        self.bidirectional_kg_obj.setup_graph()

        self.assertEqual(set(self.kg_obj.adj_lists[0]), set([1, 2]))
        self.assertEqual(set(self.kg_obj.adj_lists[1]), set([2, 0]))
        self.assertEqual(set(self.kg_obj.adj_lists[2]), set([1, 0]))

    def test_rw_adj_lists(self):
        self.bidirectional_kg_obj.run_random_walk()

        self.assertNotEqual(self.bidirectional_kg_obj.rw_adj_lists, None)

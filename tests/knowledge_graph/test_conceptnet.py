import unittest
import torch

from allennlp.common.params import Params
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG


class TestConceptNetKG(unittest.TestCase):
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
        relations = [
            "/r/IsA",
            "/r/RelatedTo",
        ]
        # (u, r, v)
        edges = [
            (
                0,
                0,
                1,
            ),
            (
                0,
                1,
                2,
            ),
            (
                1,
                0,
                2,
            ),
        ]
        features = torch.randn((3, 10))
        self.kg_obj = ConceptNetKG(
            nodes,
            features,
            edges,
            relations,
            params,
        )

        self.bidirectional_kg_obj = ConceptNetKG(
            nodes,
            features,
            edges,
            relations,
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

        self.assertEqual(self.kg_obj.adj_lists[0], set([(1, 0), (2, 1)]))
        self.assertEqual(self.kg_obj.adj_lists[1], set([(2, 0)]))

    def test_bidrectional_adj_lists(self):
        # test graph
        self.kg_obj.setup_graph(bidirectional=True)

        self.assertEqual(set(self.kg_obj.adj_lists[0]), set([(1, 0), (2, 1)]))
        self.assertEqual(set(self.kg_obj.adj_lists[1]), set([(2, 0), (0, 0)]))
        self.assertEqual(set(self.kg_obj.adj_lists[2]), set([(1, 0), (0, 1)]))

        self.bidirectional_kg_obj.setup_graph()

        self.assertEqual(set(self.kg_obj.adj_lists[0]), set([(1, 0), (2, 1)]))
        self.assertEqual(set(self.kg_obj.adj_lists[1]), set([(2, 0), (0, 0)]))
        self.assertEqual(set(self.kg_obj.adj_lists[2]), set([(1, 0), (0, 1)]))

    def test_rw_adj_lists(self):
        self.bidirectional_kg_obj.run_random_walk()

        self.assertNotEqual(self.bidirectional_kg_obj.rw_adj_lists, None)

    def test_load_from_disk(self):
        kg = self.kg_obj.load_from_disk(
            "tests/test_data/subgraphs/snips/train_graph/"
        )
        self.assertNotEqual(kg.adj_lists, None)
        self.assertNotEqual(kg.rw_adj_lists, None)

import unittest
import torch

from allennlp.common.params import Params
from torch.jit import Error
from zsl_kg.knowledge_graph.kg import KG


class TestKG(unittest.TestCase):
    def setUp(
        self,
    ):
        """creates an instance of KG with sample data."""
        params = Params({"embedding_path": ""})
        nodes = [
            "cat",
            "dog",
            "elephant",
        ]
        edges = [
            (0, 1),
            (0, 2),
        ]
        features = torch.randn((3, 10))
        self.kg_obj = KG(
            nodes,
            features,
            edges,
            params=params,
        )
        # self.kg_empty_param = KG(
        #     nodes,
        #     edges,
        # )
        self.dir_path = "tests/save_data/kg/"

    def test_to(
        self,
    ):
        """test the .to(device) function"""
        self.assertEqual(
            self.kg_obj.to(torch.device("cpu")),
            True,
        )

    def test_cuda(
        self,
    ):
        """
        test the .cuda() function
        """
        # Cannot be tested on a cpu machine
        pass

    def test_save_to_disk(
        self,
    ):
        """test the .save_to_disk function"""
        self.assertEqual(
            self.kg_obj.save_to_disk(self.dir_path),
            self.dir_path,
        )

    def test_nodes(
        self,
    ):
        """test the .nodes property"""
        self.assertEqual(
            self.kg_obj.nodes,
            [
                "cat",
                "dog",
                "elephant",
            ],
        )

    def test_get_node_ids(
        self,
    ):
        """test the .edges property"""
        self.assertEqual(
            self.kg_obj.get_node_ids(
                [
                    "cat",
                    "dog",
                ]
            ),
            [0, 1],
        )

        with self.assertRaises(Exception):
            self.kg_obj.get_node_ids(
                [
                    "cat",
                    "seal",
                ]
            )

    def test_edges(
        self,
    ):
        """test the .edges property"""
        self.assertEqual(
            self.kg_obj.edges,
            [
                (
                    0,
                    1,
                ),
                (
                    0,
                    2,
                ),
            ],
        )

    def test_load_from_disk(self):
        kg = self.kg_obj.load_from_disk(
            "tests/test_data/subgraphs/snips/train_graph/"
        )
        self.assertNotEqual(kg.adj_lists, None)
        self.assertNotEqual(kg.rw_adj_lists, None)

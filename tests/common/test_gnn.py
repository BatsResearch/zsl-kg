### unit tests for common gnn

import unittest

import torch
from allennlp.common.params import Params
from zsl_kg.common.graph import (
    GraphFeature,
    NeighSampler,
    pad_tensor,
    switch_subgraph_ids,
    get_individual_tokens,
)
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG


class TestCommonGNN(unittest.TestCase):
    def setUp(
        self,
    ):
        self.adj_nodes_list = {0: [1, 2], 1: [0], 2: [0]}

    def test_pad_tensor(self):
        """Testing pad tensor"""
        nodes_list = [self.adj_nodes_list[i] for i in range(3)]
        exp_padded_tensor = torch.tensor([[1, 2], [0, 0], [0, 0]])
        exp_mask = torch.tensor([[1, 1], [1, 0], [1, 0]])
        wrong_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 0, 0]])

        padt, mask = pad_tensor(adj_nodes_list=nodes_list)

        self.assertEqual(torch.equal(exp_padded_tensor, padt), True)
        self.assertEqual(torch.equal(exp_mask, mask), True)

        self.assertEqual(torch.equal(mask, wrong_mask), False)

    def test_switch_subgraph_ids(self):
        """Test switch subgraph ids with a mapping"""
        mapping = {0: 100, 1: 101, 2: 102}
        nodes_list = [self.adj_nodes_list[i] for i in range(3)]
        exp_nodes_list = [[101, 102], [100], [100]]

        self.assertEqual(
            exp_nodes_list, switch_subgraph_ids(nodes_list, mapping)
        )

    def test_get_individual_tokens(
        self,
    ):
        """tests the split individual tokens"""
        test_examples = [
            ["/c/en/elephant", ["elephant"]],
            ["/c/en/hand_ball", ["hand", "ball"]],
            ["/c/en/football_cup/n", ["football", "cup"]],
            ["/c/en/football_cup/n/wn/article", ["football", "cup"]],
        ]

        for _input, exp_output in test_examples:
            output = get_individual_tokens(_input)
            [
                self.assertEqual(output[i], exp_output[i])
                for i in range(len(output))
            ]


class TestNeighSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.graph_sampler_topk = NeighSampler(5, mode="topk")
        self.graph_sampler_uniform = NeighSampler(1, mode="uniform")
        self.graph_sampler_all = NeighSampler(-1, mode="none")

        params = Params({"bidirectional": True})
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
        self.kg_obj.run_random_walk()

    def test_graph_sampler(self):
        """testing the graph sampler"""

        neigh_list = self.graph_sampler_topk.sample([0], self.kg_obj)
        self.assertEqual(2, len(neigh_list[0]))

        neigh_list = self.graph_sampler_uniform.sample([0], self.kg_obj)
        self.assertEqual(1, len(neigh_list[0]))

        neigh_list = self.graph_sampler_all.sample([0], self.kg_obj)
        self.assertEqual(2, len(neigh_list[0]))

    def test_graph_sampler_enable(self):
        """testing the graph sampler enable"""
        self.graph_sampler_topk.enable()
        self.assertEqual(self.graph_sampler_topk.sample_graph, True)

    def test_graph_sampler_disable(self):
        """testing the graph sampler disable"""
        self.graph_sampler_topk.disable()
        self.assertEqual(self.graph_sampler_topk.sample_graph, False)


class TestGraphFeature(unittest.TestCase):
    def setUp(self) -> None:
        params = Params({"bidirectional": True})
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
        self.kg_obj.run_random_walk()
        self.graph_feature = GraphFeature()

    def test_graph_feature(self):
        """Embs for features from the knowledge graph"""
        embs = self.graph_feature(torch.tensor([0, 1]), self.kg_obj)
        self.assertEqual(embs.size(0), 2)
        self.assertEqual(embs.size(1), 10)

        # TODO: add tests for features with a class

import unittest
from zsl_kg.common.graph import NeighSampler

import torch
from allennlp.common.params import Params
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG
from zsl_kg.gnn.rgcn_agg import RGCNAgg


class TestRGCNAggregator(unittest.TestCase):
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

        rgcn_args = {
            "features": None,
            "input_dim": 10,
            "output_dim": 20,
            "num_rel": 3,
            "num_basis": 2,
            "sampler": NeighSampler(-1, "none"),
            "feature_dropout": 0.1,
            "self_rel_id": 2,
            "self_loop": True,
        }

        self.graph_agg = RGCNAgg(**rgcn_args)

    def test_forward(self):
        """testing forward function from the attention aggregator"""
        features = self.graph_agg(torch.tensor([0, 1]), self.kg_obj)
        self.assertEqual(features.size(0), 2)
        self.assertEqual(features.size(1), 20)

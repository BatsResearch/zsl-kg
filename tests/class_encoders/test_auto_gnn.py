import unittest
from zsl_kg.common.graph import NeighSampler

import torch
import torch.nn as nn
from allennlp.common.params import Params
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG

from zsl_kg.class_encoders import AutoGNN
from tests.class_encoders.templates import gcn, gat, rgcn, trgcn, lstm

GNN_CONFIGS = {
    "gcn": gcn,
    "gat": gat,
    "rgcn": rgcn,
    "trgcn": trgcn,
    "lstm": lstm,
}


class TestAutoGNN(unittest.TestCase):
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
        features = torch.randn((3, 300))
        self.kg_obj = ConceptNetKG(
            nodes,
            features,
            edges,
            relations,
            params,
        )
        self.kg_obj.run_random_walk()

    # TODO: add multiple subtests
    def test_forward(self):
        """testing forward function from the attention aggregator"""

        for type, config in GNN_CONFIGS.items():
            with self.subTest(i=type):
                gnn_model = AutoGNN(config)
                features = gnn_model(torch.tensor([0, 1]), self.kg_obj)

                self.assertEqual(features.size(0), 2)
                self.assertEqual(features.size(1), 2049)

    def test_weights(self):
        gnn_model = AutoGNN(gcn)

        self.assertEqual(
            set([a for a, b in list(gnn_model.named_parameters())]),
            set(
                [
                    "conv.enc.w",
                    "conv.enc.b",
                    "conv.enc.aggregator.features.features.enc.w",
                    "conv.enc.aggregator.features.features.enc.b",
                ]
            ),
        )

    def test_named_parameters(self):
        # this test will not work when concat=True, or add_weight=True because
        # self.features will be added to the encoder, which will be part
        # of the state_dict() but not the named_parameters.
        for type, config in GNN_CONFIGS.items():
            with self.subTest(i=type):
                gnn_model = AutoGNN(config)
                self.assertEqual(
                    set([a for a in gnn_model.state_dict().keys()]),
                    set([a for a, b in list(gnn_model.named_parameters())]),
                )

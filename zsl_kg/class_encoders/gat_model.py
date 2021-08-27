import torch
import torch.nn as nn
from zsl_kg.common.graph import NeighSampler
from zsl_kg.gnn.attention_agg import AttnAggregator
from zsl_kg.gnn.combine import AttentionCombine
from zsl_kg.knowledge_graph.kg import KG


class GATConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: object,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        leaky_relu_neg_slope: float = 0.2,
        self_loop: bool = True,
        activation: object = None,
        normalize: bool = False,
    ):
        """Graph attention networks adapted from
        https://arxiv.org/abs/1710.10903.

        Args:
            input_dim (int): input dimension of the node feature
            output_dim (int): output dimension of the node feature.
            features (object): The combine function or None depending
                on the layer number.
            sampler (NeighSampler, optional):  Graph sampler for the
                knowledge graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            leaky_relu_neg_slope (float, optional): leaky relu slope
                used for attention. Defaults to 0.2.
            self_loop (bool, optional): includes a self loop of the
                node in its neighbourhood. Defaults to True.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU.. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
        """
        super().__init__()

        agg = AttnAggregator(
            features,
            input_dim,
            output_dim,
            sampler,
            feature_dropout,
            leaky_relu_neg_slope,
            self_loop,
        )

        self.enc = AttentionCombine(agg, activation, normalize)

    def forward(self, node_idx: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """
        return self.enc(node_idx, kg)

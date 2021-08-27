import torch
import torch.nn as nn
from zsl_kg.common.graph import NeighSampler
from zsl_kg.gnn.combine import Combine
from zsl_kg.gnn.lstm_agg import LSTMAggregator
from zsl_kg.knowledge_graph.kg import KG


class LSTMConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: object,
        lstm_dim: int,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        self_loop: bool = True,
        shuffle: bool = True,
        self_concat: bool = False,
        activation: object = None,
        normalize: bool = False,
    ):
        """LSTM graph convolutional networks adapted from
        https://arxiv.org/abs/1706.02216.

        Args:
            input_dim (int): input dimension of the node feature
            output_dim (int): output dimension of the node feature.
            features (object): The combine function or None depending
                on the layer number.
            lstm_dim (int): dimension of the hidden states of the lstm.
            sampler (NeighSampler, optional): sampler for the knowledge
                graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            self_loop (bool, optional): adds a self loop to the graph
                if not present. Defaults to True.
            shuffle (bool, optional): shuffles the nodes in the
                neighbourhood as LSTMs are NOT permutation invariant.
                Defaults to True.
            self_concat (bool, optional): indicates if the node should be
                concatenated with its previous feature. Defaults to False.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU.. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
        """
        super().__init__()

        agg = LSTMAggregator(
            features,
            input_dim,
            lstm_dim,
            sampler,
            feature_dropout,
            self_loop,
            shuffle,
        )

        self.enc = Combine(
            input_dim,
            output_dim,
            features,
            agg,
            self_concat,
            activation,
            normalize,
        )

    def forward(self, node_idx: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """
        return self.enc(node_idx, kg)

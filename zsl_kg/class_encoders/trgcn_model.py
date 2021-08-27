import torch
import torch.nn as nn
from zsl_kg.common.graph import NeighSampler
from zsl_kg.gnn.combine import Combine
from zsl_kg.gnn.transformer_agg import TransformerAggregator
from zsl_kg.knowledge_graph.kg import KG


class TrGCNConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: object,
        self_concat: bool = False,
        activation: object = None,
        normalize: bool = False,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        num_layers: int = 1,
        num_heads: int = 1,
        pd: int = None,
        hd: int = None,
        fh: int = None,
        maxpool: bool = False,
        dp: float = 0.1,
        self_loop: bool = True,
    ):
        """Transformer graph convolutional networks adapted from
        https://arxiv.org/abs/2006.10713.

        Args:
            input_dim (int): input dimension of the node feature
            output_dim (int): output dimension of the node feature.
            features (object): The combine function or None depending
                on the layer number.
            self_concat (bool, optional): indicates if the node
                should be concatenated with its previous feature.
                Defaults to False.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU.. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
            sampler (NeighSampler, optional):  Graph sampler for the
                knowledge graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            num_layers (int, optional): number of layers in the
                transformer. Defaults to 1.
            num_heads (int, optional): number of transformer heads.
                Defaults to 1.
            pd (int, optional): project dimension in the transformer.
                If pd is None, then pd = int(input_dim/2).
                Defaults to None.
            hd (int, optional): hidden dimension or the output
                dimension of the transformer. If hd is None, then
                hd = input_dim.Defaults to None.
            fh (int, optional): feedforward layer hidden dimension.
                If fh is None, then fh = int(input_dim/2).
                Defaults to None.
            maxpool (bool, optional): if true, the aggregator uses
                maxpool instead of mean pooling. Defaults to False.
            dp (float, optional): dropout probability in the
                transformer. Defaults to 0.1.
            self_loop (bool, optional): includes a self loop of the
                node in its neighbourhood. Defaults to True.
        """
        super().__init__()

        agg = TransformerAggregator(
            features,
            input_dim,
            sampler,
            feature_dropout,
            num_layers,
            num_heads,
            pd,
            hd,
            fh,
            maxpool,
            dp,
            self_loop,
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

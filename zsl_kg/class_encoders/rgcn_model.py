import torch
import torch.nn as nn
from zsl_kg.common.graph import NeighSampler
from zsl_kg.gnn.combine import RCGNCombine
from zsl_kg.gnn.rgcn_agg import RGCNAgg
from zsl_kg.knowledge_graph.kg import KG


class RGCNConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: object,
        num_basis: int,
        num_rel: int,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        self_loop: bool = True,
        self_rel_id: int = 0,
        add_weight: bool = False,
        activation: object = None,
        normalize: bool = False,
    ):
        """Relational graph convolutional networks adapted from
        https://arxiv.org/abs/1703.06103. Implemenents basis
        decomposition described in Equation 3.

        Args:
            input_dim (int): input dimension of the node feature
            output_dim (int): output dimension of the node feature.
            features (object): The combine function or None depending
                on the layer number.
            num_basis (int): number of basis vectors
            num_rel (int): number of relations
            sampler (NeighSampler, optional): sampler for the knowledge
                graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            self_loop (bool, optional): adds a self loop to the graph
                if not present. Defaults to True.
            self_loop_rel_id (int, optional): relation index for self
                loop. Defaults to 0.
            add_weight (bool, optional): add the self feature to the
                node features. Defaults to False.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
        """
        super().__init__()

        agg = RGCNAgg(
            features,
            input_dim,
            output_dim,
            num_basis,
            num_rel,
            sampler,
            feature_dropout,
            self_loop,
            self_rel_id,
        )

        self.enc = RCGNCombine(
            input_dim,
            output_dim,
            features,
            agg,
            add_weight,
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

import torch
from torch.functional import norm
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from zsl_kg.common.graph import GraphFeature
from zsl_kg.knowledge_graph.kg import KG


class Combine(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: object,
        aggregator: object,
        self_concat: bool = False,
        activation: object = None,
        normalize: bool = False,
    ):
        """Combine class projects the aggregated features to a new
        vector space by multiplying with a linear layer followed by
        an optional activation layer and optinal normalization.
        This is a generic combine method that is used with GCN, LSTM-
        and TrGCN.

        Args:
            input_dim (int): input dimension of the node feature
            output_dim (int): output dimension of the node feature.
            features (Object): either an object or None. This is only
                used when the model needs to concatenate its previous
                layer's feature.
            aggregator ([type]): the graph aggregators.
            self_concat (bool, optional): indicates if the node should be
            concatenated with its previous feature. Defaults to False.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
        """
        super(Combine, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator

        self.self_concat = self_concat

        if not self.self_concat:
            self.features = None
            self.w = nn.Parameter(torch.empty(input_dim, output_dim))
            self.b = nn.Parameter(torch.zeros(output_dim))
        else:
            self.features = GraphFeature(features)
            self.w = nn.Parameter(torch.empty(2 * input_dim, output_dim))
            self.b = nn.Parameter(torch.zeros(output_dim))

        init.xavier_uniform_(self.w)

        self.register_parameter("w", self.w)
        self.register_parameter("b", self.b)

        self.activation = activation
        self.normalize = normalize

    def forward(self, nodes: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """
        neigh_feats = self.aggregator(nodes, kg)
        if self.self_concat:
            self_feats = self.features(nodes, kg)
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        output = torch.mm(combined, self.w) + self.b

        if self.activation is not None:
            output = self.activation(output)

        if self.normalize:
            output = F.normalize(output)

        return output


class AttentionCombine(nn.Module):
    def __init__(
        self,
        aggregator: object,
        activation: object = None,
        normalize: bool = False,
    ):
        """Combine for the attention aggregator.

        Args:
            aggregator (object): the graph aggregator.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
        """
        super(AttentionCombine, self).__init__()
        self.aggregator = aggregator
        self.activation = activation
        self.normalize = normalize

    def forward(self, nodes: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """
        output = self.aggregator(nodes, kg)

        if self.activation is not None:
            output = self.activation(output)

        if self.normalize:
            output = F.normalize(output)
        return output


class RCGNCombine(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: object,
        aggregator: object,
        add_weight: bool = False,
        activation: object = None,
        normalize: bool = False,
    ):
        """RGCN combine class for the rgcn aggregator.

        Args:
            input_dim (int): the input dimension of the node.
            output_dim (int): the output dimension of the node.
            features (object): either an object or None. This is only
                used when the model needs to concatenate its previous
                layer's feature.
            aggregator (object): the graph aggregators.
            add_weight (bool, optional): add the self feature to the
                node features. Defaults to False.
            activation (object, optional): the activation function
                torch.nn such as ReLU or LeakyReLU. Defaults to None.
            normalize (bool, optional): uses L2 normalization if true.
                Defaults to False.
        """
        super(RCGNCombine, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.aggregator = aggregator
        self.add_weight = add_weight

        if add_weight:
            self.features = GraphFeature(features)

            self.w = nn.Parameter(torch.empty(input_dim, output_dim))
            self.b = nn.Parameter(torch.zeros(output_dim))

            init.xavier_uniform_(self.w)
            self.register_parameter("w", self.w)
            self.register_parameter("b", self.b)
        else:
            self.features = None

        self.activation = activation
        self.normalize = normalize

    def forward(self, nodes: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """
        neigh_feats = self.aggregator(nodes, kg)

        # Wh_{v}^(l-1) + a_{v}^{l}
        if self.add_weight:
            self_feats = self.features(nodes, kg)
            output = torch.mm(self_feats, self.w) + neigh_feats
        else:
            output = neigh_feats

        if self.activation is not None:
            output = self.activation(output)

        if self.normalize:
            output = F.normalize(output)

        return output

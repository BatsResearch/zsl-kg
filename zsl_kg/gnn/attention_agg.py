from zsl_kg.knowledge_graph.kg import KG
import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax
from torch.nn import init
from zsl_kg.common.graph import (
    switch_subgraph_ids,
    pad_tensor,
    NeighSampler,
    GraphFeature,
)


class AttnAggregator(nn.Module):
    def __init__(
        self,
        features: object,
        input_dim: int,
        output_dim: int,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        leaky_relu_neg_slope: float = 0.2,
        self_loop: bool = True,
    ):
        """Graph attention networks (GAT) aggregator.

        Args:
            features (Object): The combine function or None depending
                on the layer number.
            input_dim(int): Input dimensions of the node features
            output_dim (int): Output dimensions of the node features
            sampler (NeighSampler, optional): Graph sampler for the
                knowledge graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            leaky_relu_neg_slope (float, optional): leaky relu slope
                used for attention. Defaults to 0.2.
            self_loop (bool, optional): includes a self loop of the
                node in its neighbourhood. Defaults to True.
        """
        super(AttnAggregator, self).__init__()

        self.features = GraphFeature(features)
        if sampler is None:
            self.sampler = NeighSampler(-1, mode="none")
        else:
            self.sampler = sampler

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        init.xavier_uniform_(self.proj.weight)

        self.feature_dropout = nn.Dropout(feature_dropout)

        self.attn_src = nn.Linear(output_dim, 1, bias=False)
        self.attn_dst = nn.Linear(output_dim, 1, bias=False)

        self.leaky_relu_neg_slope = leaky_relu_neg_slope
        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_neg_slope)
        self.self_loop = self_loop

    def forward(self, nodes: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """

        _neighs = self.sampler.sample([int(n) for n in nodes], kg)

        samp_neighs = []
        for i, adj_list in enumerate(_neighs):
            samp_neighs.append(set([node_tuple[0] for node_tuple in adj_list]))
            if self.self_loop:
                samp_neighs[i].add(int(nodes[i]))

        unique_nodes_list = list(set.union(*samp_neighs))

        # get the unique nodes
        unique_nodes = list(set(unique_nodes_list))
        node_to_emb_idx = {n: i for i, n in enumerate(unique_nodes)}
        unique_nodes_tensor = torch.tensor(unique_nodes).type_as(nodes)

        embed_matrix = self.features(unique_nodes_tensor, kg)
        embed_matrix = self.feature_dropout(embed_matrix)

        # get new features
        embed_matrix_prime = self.proj(embed_matrix)

        to_feats = torch.empty(len(samp_neighs), self.input_dim).type_as(
            embed_matrix
        )
        modified_adj_nodes = switch_subgraph_ids(samp_neighs, node_to_emb_idx)

        #
        padded_tensor, mask = pad_tensor(modified_adj_nodes)
        # sending padded tensor
        padded_tensor = padded_tensor.type_as(nodes)
        mask = mask.type_as(nodes)

        dst_nodes = []
        max_length = mask.size(1)
        for node in nodes:
            dst_nodes.append([node_to_emb_idx[int(node)]] * max_length)

        dst_tensor = torch.tensor(dst_nodes).type_as(nodes)

        # embed matrix
        neigh_feats = embed_matrix_prime[padded_tensor]
        dst_feats = embed_matrix_prime[dst_tensor]

        # attention
        dst_attn = self.leaky_relu(self.attn_dst(dst_feats))
        neigh_attn = self.leaky_relu(self.attn_src(neigh_feats))

        edge_attn = dst_attn + neigh_attn

        attn = masked_softmax(edge_attn, mask.unsqueeze(-1), dim=1)

        # multiply attention
        to_feats = torch.sum(attn * neigh_feats, dim=1)

        return to_feats

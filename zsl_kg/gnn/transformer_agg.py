import torch
import torch.nn as nn
from allennlp.modules.seq2seq_encoders.stacked_self_attention import (
    StackedSelfAttentionEncoder,
)
from allennlp.nn.util import masked_max, masked_mean
from zsl_kg.common.graph import (
    pad_tensor,
    switch_subgraph_ids,
    GraphFeature,
    NeighSampler,
)
from zsl_kg.knowledge_graph.kg import KG


class TransformerAggregator(nn.Module):
    def __init__(
        self,
        features: object,
        input_dim: int,
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
        """Transformer graph aggregator adapted from
        https://arxiv.org/abs/2006.10713.

        Args:
            features (object): The combine function or None depending
                on the layer number.
            input_dim (int): input dimension of the node feature.
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
        super(TransformerAggregator, self).__init__()

        self.features = GraphFeature(features)
        if sampler is None:
            self.sampler = NeighSampler(-1, mode="none")
        else:
            self.sampler = sampler

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.proj_dim = pd or int(input_dim / 2)
        self.hidden_dim = hd or input_dim
        self.ff_hidden = fh or int(input_dim / 2)
        self.maxpool = maxpool
        self.self_loop = self_loop

        self.input_dim = input_dim
        self.feature_dropout = nn.Dropout(feature_dropout)

        self.attention = StackedSelfAttentionEncoder(
            input_dim,
            hidden_dim=self.hidden_dim,
            projection_dim=self.proj_dim,
            feedforward_hidden_dim=self.ff_hidden,
            num_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            use_positional_encoding=False,
            attention_dropout_prob=dp,
        )

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

        unique_nodes_list = []

        unique_nodes_list = list(set.union(*samp_neighs))

        # get the unique nodes
        unique_nodes = list(set(unique_nodes_list))
        node_to_emb_idx = {n: i for i, n in enumerate(unique_nodes)}
        unique_nodes_tensor = torch.tensor(unique_nodes).type_as(nodes)

        embed_matrix = self.features(unique_nodes_tensor, kg)
        embed_matrix = self.feature_dropout(embed_matrix)

        to_feats = torch.empty(len(samp_neighs), self.input_dim).type_as(
            embed_matrix
        )

        modified_adj_nodes = switch_subgraph_ids(samp_neighs, node_to_emb_idx)

        padded_tensor, mask = pad_tensor(modified_adj_nodes)
        # sending padded tensor
        padded_tensor = padded_tensor.type_as(nodes)
        mask = mask.type_as(nodes)

        # embed matrix
        neigh_feats = embed_matrix[padded_tensor]

        attn_feats = self.attention(neigh_feats, mask)
        if self.maxpool:
            to_feats = masked_max(attn_feats, mask.unsqueeze(-1), dim=1)
        else:
            to_feats = masked_mean(attn_feats, mask.unsqueeze(-1), dim=1)

        return to_feats

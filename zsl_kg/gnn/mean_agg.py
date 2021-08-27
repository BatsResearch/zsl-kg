import torch
import torch.nn as nn

from zsl_kg.common.graph import GraphFeature, NeighSampler
from zsl_kg.knowledge_graph.kg import KG


class MeanAggregator(nn.Module):
    def __init__(
        self,
        features: object,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        self_loop: bool = True,
    ):
        """Mean aggregator from Inductive Representation Learning on
        Large Graphs.

        Args:
            features (object): The combine function or None depending
                on the layer number.
            sampler (NeighSampler, optional): Graph sampler for the
                knowledge graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            self_loop (bool, optional): includes a self loop of the
                node in its neighbourhood. Defaults to True.
        """
        super(MeanAggregator, self).__init__()

        self.features = GraphFeature(features)

        if sampler is None:
            self.sampler = NeighSampler(-1, mode="none")
        else:
            self.sampler = sampler
        self.feature_dropout = nn.Dropout(feature_dropout)
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

        unique_nodes_list = sorted(list(set.union(*samp_neighs)))

        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [
            unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh
        ]
        row_indices = [
            i
            for i in range(len(samp_neighs))
            for j in range(len(samp_neighs[i]))
        ]
        mask[row_indices, column_indices] = 1
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh.clamp(1e-8))

        node_tensor = torch.tensor(unique_nodes_list).type_as(nodes)
        embed_matrix = self.features(node_tensor, kg)
        embed_matrix = self.feature_dropout(embed_matrix)
        mask = mask.type_as(embed_matrix)

        to_feats = mask.mm(embed_matrix)

        return to_feats

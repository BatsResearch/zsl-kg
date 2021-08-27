from collections import Counter

import torch
import torch.nn as nn
from torch.nn import init
from zsl_kg.common.graph import GraphFeature, NeighSampler
from zsl_kg.knowledge_graph.kg import KG


class RGCNAgg(nn.Module):
    def __init__(
        self,
        features: object,
        input_dim: int,
        output_dim: int,
        num_basis: int,
        num_rel: int,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        self_loop: bool = True,
        self_rel_id: int = 0,
    ):
        """Relation aggregators (RGCN) from Modeling Relational Data
        with Graph Convolutional Networks. Implemenents basis
        decomposition described in Equation 3.

        Args:
            features (Object): The combine object or None.
            input_dim (int): input dimension of the node feature.
            output_dim (int): output dimension of the node feature.
            num_basis (int): number of basis vectors
            num_rel (int): number of relations
            sampler (NeighSampler, optional): sampler for the knowledge
                graph. Defaults to None.
            feature_dropout (float, optional): dropout for the node
                features. Defaults to 0.5.
            self_loop (bool, optional): adds a self loop to the graph
                if not present. Defaults to True.
            self_rel_id (int, optional): relation index for self
                loop. Defaults to 0.
        """
        super(RGCNAgg, self).__init__()

        self.features = GraphFeature(features)

        if sampler is None:
            self.sampler = NeighSampler(-1, mode="none")
        else:
            self.sampler = sampler

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.num_rel = num_rel
        self.rel_coef = nn.Parameter(torch.empty(num_rel, num_basis))
        self.rel_w = nn.Parameter(
            torch.empty(input_dim, output_dim, num_basis)
        )
        init.xavier_uniform_(self.rel_coef)
        init.xavier_uniform_(self.rel_w)

        self.feature_dropout = nn.Dropout(feature_dropout)
        self.self_loop = self_loop
        self.self_rel_id = self_rel_id

        self.register_parameter("rel_coef", self.rel_coef)
        self.register_parameter("rel_w", self.rel_w)

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
            samp_neighs.append(
                set(
                    [(node_tuple[0], node_tuple[1]) for node_tuple in adj_list]
                )
            )
            samp_neighs[i].add((int(nodes[i]), self.self_rel_id))

        unique_nodes_list = []
        unique_node_rel_list = []

        for adj_list in samp_neighs:
            adj_nodes, _ = zip(*adj_list)
            unique_nodes_list.extend(adj_nodes)
            unique_node_rel_list.extend(list(adj_list))

        # get the unique nodes
        unique_nodes = list(set(unique_nodes_list))
        node_to_emb_idx = {n: i for i, n in enumerate(unique_nodes)}
        unique_nodes_tensor = torch.tensor(unique_nodes).type_as(nodes)

        embed_matrix = self.features(unique_nodes_tensor, kg)
        embed_matrix = self.feature_dropout(embed_matrix)

        unique_node_rel_list = list(set(unique_node_rel_list))
        node_rel_idx_mapping = {
            tuple(n_r): i for i, n_r in enumerate(unique_node_rel_list)
        }
        node, rel = zip(*unique_node_rel_list)
        node = [node_to_emb_idx[n] for n in list(node)]

        rel = list(rel)
        node_embs = embed_matrix[node]
        # associate property
        # a(bc) = (ab)c
        # alpa
        b = torch.matmul(node_embs, self.rel_w.permute(1, 0, 2))
        a = self.rel_coef[rel]
        node_rel_matrix = torch.bmm(
            a.unsqueeze(1), b.permute(1, 2, 0)
        ).squeeze(1)

        # get counts per relation in row
        norm = []
        for samp_neigh in samp_neighs:
            # for n_r in samp_neigh:
            n, r = zip(*samp_neigh)
            r_count = Counter(r)
            for node, rel in samp_neigh:
                norm.append(1 / r_count[rel])

        mask = torch.zeros(
            len(samp_neighs), len(unique_node_rel_list)
        ).type_as(embed_matrix)
        column_indices = [
            node_rel_idx_mapping[n_r]
            for samp_neigh in samp_neighs
            for n_r in samp_neigh
        ]
        row_indices = [
            i
            for i in range(len(samp_neighs))
            for _ in range(len(samp_neighs[i]))
        ]
        mask[row_indices, column_indices] = torch.Tensor(norm).type_as(
            embed_matrix
        )

        to_feats = mask.mm(node_rel_matrix)

        return to_feats

import random

import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import (
    PytorchSeq2VecWrapper,
)
from zsl_kg.common.graph import (
    GraphFeature,
    NeighSampler,
    pad_tensor,
    switch_subgraph_ids,
)
from zsl_kg.knowledge_graph.kg import KG


class LSTMAggregator(nn.Module):
    def __init__(
        self,
        features: object,
        input_dim: int,
        lstm_dim: int,
        sampler: NeighSampler = None,
        feature_dropout: float = 0.5,
        self_loop: bool = True,
        shuffle: bool = True,
    ):
        """LSTM aggregator from Inductive Representation Learning on
        Large Graphs.

        Args:
            features (Object): The combine object or None.
            input_dim (int): input dimension of the node feature
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
        """
        super(LSTMAggregator, self).__init__()

        self.features = GraphFeature(features)

        if sampler is None:
            self.sampler = NeighSampler(-1, mode="none")
        else:
            self.sampler = sampler

        self.input_dim = input_dim
        self.lstm_dim = lstm_dim

        self.lstm = PytorchSeq2VecWrapper(
            nn.LSTM(
                self.input_dim,
                self.lstm_dim,
                batch_first=True,
                bidirectional=False,
            )
        )
        self.self_loop = self_loop
        self.shuffle = shuffle
        self.feature_dropout = nn.Dropout(feature_dropout)

    def forward(self, nodes: torch.tensor, kg: KG):
        """Forward function for the lstm aggregator.

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

        adj_nodes_list = []
        unique_nodes_list = []
        for adj_list in samp_neighs:
            if self.shuffle:
                adj_list = list(adj_list)
                random.shuffle(adj_list)

            unique_nodes_list.extend(adj_list)
            adj_nodes_list.append(adj_list)

        unique_nodes = list(set(unique_nodes_list))

        idx_mapping = {n: i for i, n in enumerate(unique_nodes)}
        unique_nodes_tensor = torch.tensor(unique_nodes).type_as(nodes)
        embs_tensor = self.features(unique_nodes_tensor, kg)
        embs_tensor = self.feature_dropout(embs_tensor)

        # adding a zero tensor for padding
        modified_adj_nodes = switch_subgraph_ids(adj_nodes_list, idx_mapping)

        padded_tensor, mask = pad_tensor(modified_adj_nodes)
        padded_tensor = padded_tensor.type_as(nodes)
        mask = mask.type_as(nodes)
        padded_embs = embs_tensor[padded_tensor]

        # create mask
        hidden_states = self.lstm(padded_embs, mask)

        return hidden_states

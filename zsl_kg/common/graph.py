from typing import Generic
import random
import torch
import torch.nn as nn
import re

from zsl_kg.knowledge_graph.kg import KG


def pad_tensor(adj_nodes_list):
    """Function pads the neighbourhood nodes before passing through the
    aggregator.
    Args:
        adj_nodes_list (list): the list of node neighbours
    Returns:
        tuple: one of two tensors containing the padded tensor and mask
    """
    max_len = max([len(adj_nodes) for adj_nodes in adj_nodes_list])

    padded_nodes = []
    _mask = []
    for adj_nodes in adj_nodes_list:
        x = list(adj_nodes)

        x += [0] * (max_len - len(adj_nodes))
        padded_nodes.append(x)
        _mask.append([1] * len(adj_nodes) + [0] * (max_len - len(adj_nodes)))

    # returning the mask as well
    return torch.tensor(padded_nodes), torch.tensor(_mask)


def switch_subgraph_ids(adj_nodes_list, idx_mapping):
    """function maps the node indices to new indices using a mapping
    dictionary.
    Args:
        adj_nodes_list (list): list of list containing the node ids
        idx_mapping (dict): node id to mapped node id
    Returns:
        list: list of list containing the new mapped node ids
    """
    new_adj_nodes_list = []
    for adj_nodes in adj_nodes_list:
        new_adj_nodes = []
        for node in adj_nodes:
            new_adj_nodes.append(idx_mapping[node])
        new_adj_nodes_list.append(new_adj_nodes)

    return new_adj_nodes_list


def get_individual_tokens(concept):
    """Function splits the concept to individual tokens.
    Example: "/c/en/mountain_lion/"n -> ["mountain", "lion"]
    Args:
        concept (str): the concept name from ConceptNet
    Returns:
        list: list of individual tokens in the concept
    """
    clean_concepts = re.sub(r"\/c\/[a-z]{2}\/|\/.*", "", concept)
    return clean_concepts.strip().split("_")


class NeighSampler:
    def __init__(self, num_sample: int, mode: str = "topk"):
        """Class is used to sample the node neighbours.

        Args:
            num_sample (int): the number of nodes to sample in
            each neighbourhood.
            mode (str, optional): There are three modes: topk, uniform,
            and none. Defaults to "topk".
        """
        self.num_sample = num_sample
        self.mode = mode
        self.sample_graph = True
        if mode == "none":
            self.sample_graph = False

    def enable(self, mode: str = None):
        """Class method to enable sampling. By default this is enabled.

        Args:
            mode (str, optional): sampling type. Defaults to None.
        """
        if mode is not None:
            self.mode = mode
        # if the mode is none, then the sampling is
        self.sample_graph = True

    def disable(self):
        """Class method disables topk/uniform sampling."""
        self.sample_graph = False

    def sample(self, nodes: list, kg: KG):
        """Class method samples the node neighbour for the nodes
        passed to the function. This is an important function which
        will be used in gnn aggregators.

        Args:
            nodes (list): the nodes from which we need to sample the
            node neighbours.
            kg (KG): the knowledge graph with random-walked
            probabilities.

        Raises:
            Exception: The module raises an exception if any of the
            nodes is not present in the graph.

        Returns:
            list: list of list with the sampled graph.
        """
        neighs_list = []
        for n in nodes:
            if n not in kg.rw_adj_lists:
                raise Exception(f"{n} is not present in the graph")
            neighs_list.append(kg.rw_adj_lists[n])

        if self.sample_graph:
            if self.mode == "topk":
                sampled_neighs = [
                    sorted(neighs, key=lambda x: x[-1], reverse=True)[
                        : self.num_sample
                    ]
                    if len(neighs) >= self.num_sample
                    else neighs
                    for neighs in neighs_list
                ]
            elif self.mode == "uniform":
                sampled_neighs = [
                    random.sample(neighs, self.num_sample)
                    if len(neighs) >= self.num_sample
                    else neighs
                    for neighs in neighs_list
                ]

            return sampled_neighs
        else:
            return neighs_list


class GraphFeature(nn.Module):
    def __init__(self, features=None):
        """Class regulates the features to the graph neural network.

        Args:
            features (Object, optional): Either the class encoder
            or just None. Defaults to None.
        """
        super().__init__()
        self.features = features

    def forward(self, nodes: torch.tensor, kg: KG):
        """Function to return the features for graph neural network.

        Args:
            nodes (torch.tensor): nodes from the graph of the
            shape (n).
            kg (KG): knowledge graph object.
        """
        if self.features == None:
            return kg.features[nodes, :]
        else:
            return self.features(nodes, kg)

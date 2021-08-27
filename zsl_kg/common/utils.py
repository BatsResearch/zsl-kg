"""
utils for zsl-kg
"""
import random
from collections import Counter

import numpy as np
import torch
from allennlp.common import Tqdm
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile


def random_walk(
    adj_lists: dict, current_node: int, current_list: list, k: int
):
    """The function computes the random walk over the graph

    Arguments:
        adj_lists {dict} -- dict containing lists of neighs
        current_node {int} -- the node id (usually indicates the starting point)
        current_list {list} -- the path walked so far
        k {int} -- the number of steps remaining

    Returns:
        list -- the list of transitions
    """
    if k == 0:
        return current_list
    else:
        if current_node not in adj_lists:
            raise Exception(
                "The knowledge graph is directed. Use "
                'Params({"bidirectional": True}) when initializing '
                "the knowledge graph."
            )
        node_rel = random.sample(adj_lists[current_node], 1)[0]
        if type(node_rel) == int or type(node_rel) == np.int64:
            node = node_rel
        else:
            node = node_rel[0]
        current_list.append(node)
        return random_walk(adj_lists, node, current_list, k - 1)


def run_random_walk(
    nodes: list,
    adj_lists: dict,
    relations: bool = False,
    k: int = 20,
    n: int = 10,
    seed: int = 0,
):
    """run random walk for all the nodes in entire adj_list. The function does not
    add any self loop.

    Args:
        nodes (list): list of node strings.
        adj_lists (dict): dictionary containing the adj list for the nodes
            in the graph
        relations (bool): indicates if the adj_lists contains relations
        k (int): length of the random walk
        n (int): number of random walk restarts
        seed (int, optional): random seed value. Defaults to 0.

    Returns:
        dict: random walk with hitting probability
    """
    random.seed(seed)

    rw_adj_lists = {}
    for index in Tqdm.tqdm(range(len(nodes))):
        transitions = []
        if index not in adj_lists:
            # TODO: show warning
            continue

        for i in range(n):
            transitions += random_walk(adj_lists, index, [], k)

        # filter
        counts = Counter(transitions)
        if relations:
            nodes = set([neigh for neigh, rel in adj_lists[index]])
        else:
            nodes = adj_lists[index]

        neigh_counts = dict(
            [
                (neigh, count)
                for neigh, count in counts.items()
                if neigh in nodes
            ]
        )

        #
        if relations:
            for neigh, rel in adj_lists[index]:
                if neigh not in neigh_counts:
                    neigh_counts[neigh] = 0

            # add smoothing
            for neigh, rel in adj_lists[index]:
                neigh_counts[neigh] += 1
        else:
            for neigh in adj_lists[index]:
                if neigh not in neigh_counts:
                    neigh_counts[neigh] = 0

            # add smoothing
            for neigh in adj_lists[index]:
                neigh_counts[neigh] += 1

        # hitting probability
        total = sum([count for neigh, count in neigh_counts.items()])

        hit_prob_dict = dict(
            [
                (neigh, count / total * 1.0)
                for neigh, count in neigh_counts.items()
            ]
        )

        #
        if relations:
            rw_adj_lists[index] = [
                (neigh, rel, hit_prob_dict[neigh])
                for neigh, rel in adj_lists[index]
            ]
        else:
            rw_adj_lists[index] = [
                (neigh, hit_prob_dict[neigh]) for neigh in adj_lists[index]
            ]

    return rw_adj_lists

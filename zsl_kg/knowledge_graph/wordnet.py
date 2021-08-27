import re
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import os

from .kg import KG
from zsl_kg.common.utils import run_random_walk
from allennlp.common.params import Params


class WordNetKG(KG):
    def __init__(
        self,
        nodes: list,
        features: torch.Tensor,
        edges: list,
        params: Params = None,
    ):
        """__init__ class with relations.

        Args:
            nodes (list): list of node names
            features (torch.Tensor): the initial node features
            edges (list): list of edges (u, v)
            params (Params, optional): Contains information
                related to the random-walk. Defaults to None.
        """
        super().__init__(nodes, features, edges, params=params)
        self.bidirectional = params.pop("bidirectional", default=False)
        # random walk params
        self.k = params.pop("rw.k", default=None)
        self.n = params.pop("rw.n", default=None)
        self.seed = params.pop("rw.seed", default=None)

    def setup_graph(self, bidirectional=False):
        """This function is used to run the conversion from the list
        of edges to adj_lists, depending on the graph.

        Args:
            bidirectional (bool, optional): makes the edges
            as bidirectional. Defaults to False.
        """

        if self.adj_lists is None:
            edges_df = pd.DataFrame(self.edges, columns=["start_id", "end_id"])
            edges_df.drop_duplicates()

            if bidirectional or self.bidirectional:
                opp_edges = edges_df.copy()
                opp_edges[["start_id", "end_id"]] = edges_df[
                    ["end_id", "start_id"]
                ]
                edges_df = pd.concat((edges_df, opp_edges), ignore_index=True)
                edges_df.drop_duplicates()

            # create adj lists
            self.adj_lists = (
                edges_df[["start_id", "end_id"]]
                .set_index("start_id")
                .apply(int, 1)
                .groupby(level=0)
                .agg(lambda x: set(x.values))
                .to_dict()
            )

    def run_random_walk(self):
        """This function runs random walk over the graph. The length
        of the walk k and the number of restarts n are obtained from
        params"""
        if self.adj_lists is None:
            self.setup_graph()

        if self.rw_adj_lists is None:
            # run the random walk
            rw_params = {
                "nodes": self.nodes,
                "adj_lists": self.adj_lists,
            }

            for p in [self.k, self.n, self.seed]:
                if p is not None:
                    rw_params[p] = self.params[p]

            self.rw_adj_lists = run_random_walk(**rw_params)

    @staticmethod
    def load_from_disk(dir_path: str):
        """Function to load the graph from disk

        Args:
            dir_path (str): directory path

        Returns:
            KG: the knowledge graph
        """
        params = pickle.load(open(os.path.join(dir_path, "params.pkl"), "rb"))
        data = torch.load(os.path.join(dir_path, "graph_data.pt"))

        kg = WordNetKG(
            data["nodes"],
            data["features"],
            data["edges"],
            params,
        )

        kg.adj_lists = data["adj_lists"]
        kg.rw_adj_lists = data["rw_adj_lists"]

        return kg

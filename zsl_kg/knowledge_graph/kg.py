import os
import pickle
import torch
from allennlp.common.params import Params


class KG:
    def __init__(
        self,
        nodes: list,
        features: torch.Tensor,
        edges: list,
        relations: list = None,
        params: Params = None,
    ):
        """Generic KG class.

        Args:
            nodes (list): list of node names
            features (torch.Tensor): the initial node features
            edges (list): list of edges (u, r, v) or (u, v)
            relations (list, optional): list of relation names. Defaults to None.
            params (Params, optional): Contains information
                related to the random-walk. Defaults to None.
        """
        self.nodes = nodes
        self.features = features

        self.edges = edges
        self.relations = relations
        self.adj_lists = None
        self.rw_adj_lists = None
        if params is None:
            self.params = Params({})
        else:
            self.params = params

    def setup_graph(self):
        """This function is used to run the conversion from the list
        of edges to adj_lists, depending on the graph.
        """
        NotImplementedError()

    def run_random_walk(self):
        """This function runs random walk over the graph."""
        NotImplementedError()

    def to(self, device: torch.DeviceObjType):
        """Functions moves the node features
        to the cuda/cpu device.

        Returns:
            bool: returns True if successful
        """
        self.features = self.features.to(device)
        return True

    def cuda(self):
        """Functions moves the node features to cuda.

        Returns:
            bool: returns True if successful
        """
        self.features.cuda()
        return True

    def get_node_ids(self, nodes: list):
        """Function gets the node ids for the list
        of nodes. This is typically useful when you
        want to get the vector representation of the
        node in the knowledge graph.

        Args:
            nodes (list): list of node names in the graph.

        Raises:
            Exception: raises an exception if the node is
                not present in the knowledge graph.

        Returns:
            list: node ids of the nodes in the order
                provided.
        """
        node_ids = []
        node_to_idx = dict(
            [(node, idx) for idx, node in enumerate(self.nodes)]
        )
        for node in nodes:
            if node not in node_to_idx:
                raise Exception(f"{node} not found in the knowledge graph.")
            node_ids.append(node_to_idx[node])

        return node_ids

    def save_to_disk(self, dir_path: str):
        """Function saves the kg data to disk in a directory.

        Args:
            dir_path (str): directory path where the kg
                data will be saved
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save(
            {
                "nodes": self.nodes,
                "features": self.features.cpu(),
                "edges": self.edges,
                "relations": self.relations,
                "adj_lists": self.adj_lists,
                "rw_adj_lists": self.rw_adj_lists,
            },
            os.path.join(dir_path, "graph_data.pt"),
        )

        with open(os.path.join(dir_path, "params.pkl"), "wb+") as fp:
            pickle.dump(self.params, fp)

        return dir_path

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

        kg = KG(
            data["nodes"],
            data["features"],
            data["edges"],
            data["relations"],
            params,
        )

        kg.adj_lists = data["adj_lists"]
        kg.rw_adj_lists = data["rw_adj_lists"]

        return kg

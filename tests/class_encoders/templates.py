## config templates
import unittest
from zsl_kg.common.graph import NeighSampler

import torch
import torch.nn as nn
from allennlp.common.params import Params
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG
from zsl_kg.gnn.mean_agg import MeanAggregator

## Configs for AutoGNN for Object classification

gcn = {
    "input_dim": 300,
    "output_dim": 2049,
    "type": "gcn",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 2048,
            "activation": nn.LeakyReLU(0.2),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 2048,
            "output_dim": 2049,
            "activation": None,
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

gat = {
    "input_dim": 300,
    "output_dim": 2049,
    "type": "gat",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 2048,
            "activation": nn.LeakyReLU(0.2),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 2048,
            "output_dim": 2049,
            "activation": None,
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

rgcn = {
    "input_dim": 300,
    "output_dim": 2049,
    "type": "rgcn",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 2048,
            "activation": nn.LeakyReLU(0.2),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
            "num_basis": 1,
            "num_rel": 50 + 1,
            "self_rel_id": 50,
        },
        {
            "input_dim": 2048,
            "output_dim": 2049,
            "activation": None,
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
            "num_basis": 1,
            "num_rel": 50 + 1,
            "self_rel_id": 50,
        },
    ],
}

lstm = {
    "input_dim": 300,
    "output_dim": 2049,
    "type": "lstm",
    "gnn": [
        {
            "input_dim": 300,
            "lstm_dim": 300,
            "output_dim": 2048,
            "activation": nn.LeakyReLU(0.2),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 2048,
            "lstm_dim": 2048,
            "output_dim": 2049,
            "activation": None,
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

trgcn = {
    "input_dim": 300,
    "output_dim": 2049,
    "type": "trgcn",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 2048,
            "activation": nn.LeakyReLU(0.2),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
            "fh": 100,
        },
        {
            "input_dim": 2048,
            "output_dim": 2049,
            "activation": None,
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

import os
from typing import Text
import torch
import unittest

import torch.nn as nn
import torch.optim as optim
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from zsl_kg.class_encoders.auto_gnn import AutoGNN
from zsl_kg.example_encoders.text_encoder import TextEncoder
from zsl_kg.data.snips import SnipsDataset
from allennlp.data.iterators import BasicIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from zsl_kg.common.graph import NeighSampler
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG
from allennlp.common.tqdm import Tqdm


class BiLinearModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        example_encoder: object,
        class_encoder: object,
        joint_dim: int,
        bias: bool = False,
    ):
        super().__init__(vocab)
        self.example_encoder = example_encoder
        self.class_encoder = class_encoder

        self.text_joint = nn.Linear(
            self.example_encoder.output_dim, joint_dim, bias=bias
        )
        self.class_joint = nn.Linear(
            self.class_encoder.output_dim, joint_dim, bias=bias
        )

    def forward(self, batch, node_idx, kg):
        encoder_out = self.example_encoder(batch)
        text_rep = self.text_joint(encoder_out)

        # get label representation
        class_out = self.class_encoder(node_idx, kg)
        class_rep = self.class_joint(class_out)

        logits = torch.matmul(text_rep, class_rep.t())

        return logits


class TestIntentClassification(unittest.TestCase):
    def setUp(
        self,
    ):
        label_maps = {
            "train": ["weather", "music", "restaurant"],
            "dev": ["search", "movie"],
            "test": ["book", "playlist"],
        }

        data_path = "tests/test_data/datasets/snips/"
        datasets = []
        for split in ["train", "dev", "test"]:
            labels = label_maps[split]
            label_to_idx = dict(
                [(label, idx) for idx, label in enumerate(labels)]
            )

            reader = SnipsDataset(label_to_idx)
            path = os.path.join(data_path, f"{split}.txt")
            _dataset = reader.read(path)
            datasets.append(_dataset)

        self.train_dataset, self.dev_dataset, self.test_dataset = datasets
        vocab = Vocabulary.from_instances(
            self.train_dataset + self.dev_dataset + self.test_dataset
        )

        # create the iterator
        self.iterator = BasicIterator(batch_size=32)
        self.iterator.index_with(vocab)

        print("Loading GloVe...")
        # token embed
        token_embed_path = os.path.join(data_path, "word_emb.pt")
        token_embedding = torch.load(token_embed_path)

        print("word embeddings created...")
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

        # create the text encoder
        print("Loading the text encoder...")
        self.example_encoder = TextEncoder(word_embeddings, 300, 32, 20)

        trgcn = {
            "input_dim": 300,
            "output_dim": 64,
            "type": "trgcn",
            "gnn": [
                {
                    "input_dim": 300,
                    "output_dim": 64,
                    "activation": nn.ReLU(),
                    "normalize": True,
                    "sampler": NeighSampler(100, mode="topk"),
                    "fh": 100,
                },
                {
                    "input_dim": 64,
                    "output_dim": 64,
                    "activation": nn.ReLU(),
                    "normalize": True,
                    "sampler": NeighSampler(50, mode="topk"),
                },
            ],
        }

        self.class_encoder = AutoGNN(trgcn)

        self.train_graph = ConceptNetKG.load_from_disk(
            "tests/test_data/subgraphs/snips/train_graph"
        )
        node_to_idx = dict(
            [(node, idx) for idx, node in enumerate(self.train_graph.nodes)]
        )
        #
        self.train_nodes = torch.tensor(
            [
                node_to_idx[node]
                for node in [
                    "/c/en/weather",
                    "/c/en/music",
                    "/c/en/restaurant",
                ]
            ]
        )

        self.model = BiLinearModel(
            vocab, self.example_encoder, self.class_encoder, joint_dim=20
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-03, weight_decay=5e-04
        )
        self.loss_function = nn.CrossEntropyLoss()

    def test_intent_classification_train(self):
        self.model.train()
        total_batch_loss = 0.0
        generator_tqdm = Tqdm.tqdm(
            self.iterator(self.train_dataset, num_epochs=1, shuffle=False),
            total=self.iterator.get_num_batches(self.train_dataset),
        )

        for batch in generator_tqdm:
            self.optimizer.zero_grad()

            logits = self.model(
                batch["sentence"], self.train_nodes, self.train_graph
            )
            loss = self.loss_function(logits, batch["labels"])
            total_batch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        self.assertLessEqual(total_batch_loss, 100.0)

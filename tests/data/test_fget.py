from zsl_kg.data.fget import FineEntityTyping

import os
import torch
import unittest
import pandas as pd

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.iterators.basic_iterator import BasicIterator


class TestFineEntityTyping(unittest.TestCase):
    def setUp(
        self,
    ):
        self.dataset_path = "tests/test_data/datasets/bbn"

        train_df = pd.read_csv(
            os.path.join(self.dataset_path, "train_labels.csv")
        )
        train_labels = train_df["LABELS"].to_list()
        self.train_to_idx = dict(
            [(label, idx) for idx, label in enumerate(train_labels)]
        )

        #
        test_df = pd.read_csv(
            os.path.join(self.dataset_path, "test_labels.csv")
        )
        test_labels = test_df["LABELS"].to_list()
        all_labels = train_labels + test_labels
        self.test_to_idx = dict(
            [(label, idx) for idx, label in enumerate(all_labels)]
        )

        self.train, self.test = self.load_dataset(self.dataset_path)

        vocab = Vocabulary.from_instances(self.train + self.test)

        # instantiate iterator
        self.iterator = BasicIterator(batch_size=1000)
        self.iterator.index_with(vocab)

    def load_dataset(self, dataset_path):
        train_path = os.path.join(dataset_path, "clean_train.json")
        test_path = os.path.join(dataset_path, "clean_test.json")

        #
        train_reader = FineEntityTyping(self.train_to_idx)
        train_dataset = train_reader.read(train_path)

        #
        test_reader = FineEntityTyping(self.test_to_idx)
        test_dataset = test_reader.read(test_path)

        return train_dataset, test_dataset

    def test_batch(self):
        generator_tqdm = Tqdm.tqdm(
            self.iterator(self.train, num_epochs=1, shuffle=False),
            total=self.iterator.get_num_batches(self.train),
        )
        for batch in generator_tqdm:
            break

        self.assertEqual(batch["left_tokens"]["tokens"].size(1), 10)
        self.assertEqual(batch["right_tokens"]["tokens"].size(1), 10)

        test_label = [0] * len(self.train_to_idx)
        for label in ["/PLANT", "/PRODUCT", "/SUBSTANCE"]:
            test_label[self.train_to_idx[label]] = 1
        test_label = torch.tensor(test_label)

        self.assertEqual(torch.sum(batch["labels"][0] - test_label), 0)

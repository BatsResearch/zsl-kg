import os
from typing import Text
import torch
import unittest
from zsl_kg.data.gbu import GBU
from torch.utils.data import DataLoader


class TestGBU(unittest.TestCase):
    def setUp(
        self,
    ):
        self.awa2_train = GBU(
            "tests/test_data/datasets/awa2", [1, 2], [0, 0], stage="train"
        )
        self.awa2_test = GBU(
            "tests/test_data/datasets/awa2", [3], [1], stage="test"
        )

    def test_gbu_batch(self):
        dataloader = DataLoader(self.awa2_train, batch_size=2)
        for batch in dataloader:
            data, label = batch
            self.assertEquals(data.size(0), 2)
            self.assertEquals(label.size(0), 2)

    def test_gbu_length(self):
        self.assertEquals(len(self.awa2_train), 2)
        self.assertEquals(len(self.awa2_test), 1)

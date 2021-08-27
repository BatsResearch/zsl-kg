from typing import Iterator, Dict

import csv


from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class SnipsDataset(DatasetReader):
    def __init__(
        self,
        label_to_idx: dict = None,
    ):
        """Dataset Reader for the SNIPS dataset
        (https://arxiv.org/abs/1805.10190).

        Args:
            label_to_idx (dict, optional): label name to idx.
            Defaults to None.
        """
        super().__init__(lazy=False)
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.label_to_idx = label_to_idx

    def text_to_instance(self, text_label: list):
        """Function to convert text to instance for training
        and testing.

        Args:
            text_label (list): text and label pair from the dataset.

        Returns:
            Instance: instance from the dataset.
        """
        label = LabelField(
            self.label_to_idx[text_label[0]], skip_indexing=True
        )
        tokens = [Token(token) for token in text_label[1].split()]
        sentence_field = TextField(tokens, self.token_indexers)

        fields = {"sentence": sentence_field, "labels": label}
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Function reads dataset from the file_path.

        Args:
            file_path (str): file path of the dataset csv.

        Yields:
            Iterator[Instance]: dataset instances
        """
        with open(file_path) as f:
            text_classification_reader = csv.reader(f, delimiter="\t")
            for line in text_classification_reader:
                yield (self.text_to_instance(line))

import json

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, MultiLabelField, MetadataField


class FineEntityTyping(DatasetReader):
    def __init__(self, label_to_idx, context_size=10):
        """Dataset reader for fine-grained entity typing datasets.

        Args:
            label_to_idx (dict): label to idx mapping
            context_size (int, optional): context window size. Defaults to 10.
        """
        super().__init__(lazy=False)
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.feature_token_indexers = {
            "features": SingleIdTokenIndexer(namespace="features")
        }
        self.label_to_idx = label_to_idx
        self.context_size = context_size

    def text_to_instance(self, line: str):
        """Function converts text to instance for training
        and testing.

        Args:
            line (str): json string from the dataset.

        Returns:
            Instance: allennlp Instance
        """
        example = json.loads(line)
        start = example["start"]
        end = example["end"]
        words = example["tokens"]
        labels = example["labels"]

        start = int(start)
        end = int(end)

        left_tokens = words[:start]
        right_tokens = words[end:]

        mention_tokens = [Token(token) for token in words[start:end]]
        left_tokens = [
            Token(token)
            for token in left_tokens[
                -min(len(left_tokens), self.context_size) :
            ]
        ]
        right_tokens = [
            Token(token)
            for token in right_tokens[
                : min(len(right_tokens), self.context_size)
            ]
        ]

        # convert all of them to text fields
        mention_tokens = TextField(mention_tokens, self.token_indexers)
        left_tokens = TextField(left_tokens, self.token_indexers)
        right_tokens = TextField(right_tokens, self.token_indexers)

        label_idx = [self.label_to_idx[label] for label in labels]

        label_field = MultiLabelField(
            label_idx,
            skip_indexing=True,
            num_labels=len(self.label_to_idx),
        )

        data = {
            "mention_tokens": mention_tokens,
            "left_tokens": left_tokens,
            "right_tokens": right_tokens,
            "labels": label_field,
        }

        return Instance(data)

    def _read(self, file_path: str):
        """Function reads dataset from the file_path.

        Args:
            file_path (str): file path of the dataset json string
                lines.

        Yields:
            Iterator[Instance]: dataset instances
        """
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                yield (self.text_to_instance(line))

import torch
import torch.nn as nn

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, masked_mean

from zsl_kg.models.attention import Attention


class AttentiveNER(nn.Module):
    def __init__(
        self,
        word_embeddings: BasicTextFieldEmbedder,
        input_dim: int,
        hidden_dim: int,
        attn_dim: int,
        features_to_idx: dict = None,
        feature_dim: int = None,
        dropout: float = 0.5,
    ):
        """Class for AttentiveNER adapted from
        https://arxiv.org/abs/1604.05525 for fine-grained entity typing.
        This module has three
        tasks: (1) average the embeddings of the mention (2) compute
        the vector representation of the context with the biLSTM with
        attention, and (3) optionally, concatenate with hand-crafted
        features in the example

        Args:
            word_embeddings (BasicTextFieldEmbedder): the word
                embeddings that will be used in task.
            input_dim (int): input dimension of the biLSTM
            hidden_dim (int): output dimension of the biLSTM.
            attn_dim (int): attention dimension in the attention
                module
            features_to_idx (dict, optional): contains feature to id
                mapping. Defaults to None.
            feature_dim (int, optional): the dimension of the feature
                embedding. Defaults to None.
            dropout (float, optional): dropout added to the feature
                embedding. Defaults to 0.5.
        """
        super(AttentiveNER, self).__init__()
        self.word_embeddings = word_embeddings
        self.hidden_dim = hidden_dim
        self.left_bilstm = PytorchSeq2SeqWrapper(
            nn.LSTM(
                input_dim, hidden_dim, batch_first=True, bidirectional=True
            )
        )
        self.right_bilstm = PytorchSeq2SeqWrapper(
            nn.LSTM(
                input_dim, hidden_dim, batch_first=True, bidirectional=True
            )
        )

        self.attn = Attention(hidden_dim * 2, attn_dim)

        if features_to_idx is not None:
            self.feat_embs = nn.Embedding(
                len(features_to_idx) + 1, feature_dim, padding_idx=0
            )
            self.feat_to_idx = features_to_idx

            self.output_dim = (
                2 * hidden_dim + input_dim + feature_dim
            )  # 50 is for features
        else:
            self.feat_to_idx = None
            self.output_dim = 2 * hidden_dim + input_dim

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        mention_tokens: torch.tensor,
        left_tokens: torch.tensor,
        right_tokens: torch.tensor,
        features: torch.tensor = None,
    ):
        """The forward function for AttentiveNER. The function
        computes a vector representaiton for the mention along
        with its context.

        Args:
            mention_tokens (torch.tensor): the token ids in the
                mention.
            left_tokens (torch.tensor): the token ids to the left
                of the mention, within the context window.
            right_tokens (torch.tensor): the token ids to the right
                of the mention, within the context window.
            features (torch.tensor, optional): the token ids of the
                hand-crafted features. Defaults to None.

        Returns:
            [type]: [description]
        """
        mention_mask = get_text_field_mask(mention_tokens)
        mention_embs = self.word_embeddings(mention_tokens)
        avg_mentions = masked_mean(
            mention_embs, mention_mask.unsqueeze(-1), dim=1
        )
        avg_mentions = self.dropout(avg_mentions)

        left_mask = get_text_field_mask(left_tokens)
        left_embs = self.word_embeddings(left_tokens)
        left_hidden_states = self.left_bilstm(left_embs, left_mask)

        right_mask = get_text_field_mask(right_tokens)
        right_embs = self.word_embeddings(right_tokens)
        right_hidden_states = self.right_bilstm(right_embs, right_mask)

        full_mask = torch.cat([left_mask, right_mask], dim=1)
        full_hidden = torch.cat(
            [left_hidden_states, right_hidden_states], dim=1
        )

        attn_scores = self.attn(full_hidden, full_mask)
        context = torch.sum(full_hidden * attn_scores, dim=1)

        if self.feat_to_idx is not None:
            feat_rep = torch.sum(self.feat_embs(features), dim=1)
            feat_rep = self.dropout(feat_rep)

            mention_rep = torch.cat([avg_mentions, context, feat_rep], dim=1)
        else:
            mention_rep = torch.cat([avg_mentions, context], dim=1)

        return mention_rep

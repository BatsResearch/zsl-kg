import torch
import torch.nn as nn

from allennlp.nn.util import masked_softmax
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from zsl_kg.models.attention import Attention


class TextEncoder(nn.Module):
    def __init__(
        self,
        word_embeddings: BasicTextFieldEmbedder,
        input_dim: int,
        hidden_dim: int,
        attn_dim: int,
        bidrectional: bool = True,
        dropout: float = 0.5,
    ):
        """Class is used to learn an LSTM with attention
        or a biLSTM with attention.

        Args:
            word_embeddings (BasicTextFieldEmbedder): the word embeddings in the entire
                training set or all possible word embeddings.
            input_dim (int): input dimension of the lstm.
            hidden_dim (int): hidden state output dimension of the lstm.
            attn_dim (int): dimension of the attention in the MLP-based attention.
            bidrectional (bool, optional): if true, uses biLSTM. Defaults to True.
        """
        super(TextEncoder, self).__init__()
        self.word_embeddings = word_embeddings
        self.hidden_dim = hidden_dim
        self.bidirectional = bidrectional
        self.bilstm = PytorchSeq2SeqWrapper(
            nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        )

        self.output_dim = hidden_dim
        if self.bidirectional:
            self.output_dim *= 2

        self.attn = Attention(self.output_dim, attn_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence: torch.Tensor):
        """Forward function computes a sentence representation with
        either lstm with attention, or biLSTM with attention.

        Args:
            sentence (torch.Tensor): the token ids correspending the
                word embeddings.

        Returns:
            torch.Tensor: the sentence representation.
        """
        mask = get_text_field_mask(sentence)
        sentence_embs = self.word_embeddings(sentence)
        hidden_states = self.bilstm(sentence_embs, mask)
        attn_scores = self.attn(hidden_states, mask)
        sentence_rep = torch.sum(hidden_states * attn_scores, 1)
        sentence_rep = self.dropout(sentence_rep)
        return sentence_rep

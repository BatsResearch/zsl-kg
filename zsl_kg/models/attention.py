import torch
import torch.nn as nn

from allennlp.nn.util import masked_softmax


class Attention(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int):
        """A simple mlp-based attention module to get
        attention coefficients. The module uses tanh
        as the activation function.

        Args:
            hidden_dim (int): the input dimension of the tensor
            attn_dim (int): the attention dimension or the output
                dimension of the first layer.
        """
        super(Attention, self).__init__()
        self.linear_1 = nn.Linear(input_dim, attn_dim)
        self.linear_2 = nn.Linear(attn_dim, 1)

        self.tanh = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Forward function computes the attention coefficients
        relative to the hidden states.

        Args:
            hidden_states (torch.Tensor): the input tensors of the
                shape (batch_size, num_tensors, emb_dim)
            mask (torch.Tensor): the mask for the batch of the shape
                (batch_size, num_tensors)

        Returns:
            torch.Tensor: the attention coefficients with
                the shape (batch_size, num_tensors, 1)
        """
        lin_out = self.tanh(self.linear_1(hidden_states))
        final_out = self.linear_2(lin_out)
        masked_scores = masked_softmax(final_out, mask.unsqueeze(-1), dim=1)
        return masked_scores

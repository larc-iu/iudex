"""Pointer attention for top-down split selection.

A span of n EDUs can be split at n - 1 different points (at least one EDU on
either side). Given a query (the decoder state for the current span) and the
span's n - 1 candidate split-anchor EDU representations as keys, this scores
each candidate split.

Lives here as a self-contained `nn.Module`. Used by dmrst's recurrent pointer
decoder. It was shared with a second parser when factored out; that parser has
since been removed, so it is currently dmrst-only but kept here in anticipation
of a second pointer-decoding parser.
"""

import torch
import torch.nn as nn


class PointerAttention(nn.Module):
    """Pointer attention over candidate split positions.

    With e_k = encoder_outputs[k] and d = decoder_output:
        biaffine:    logit_k = (W1 e_k)·d + w2·e_k
        dot_product: logit_k = e_k · d

    Args:
        encoder_outputs: [n - 1, hidden_size]
        decoder_output: [hidden_size]

    Returns:
        logits: [1, n - 1]  (leading dim is F.cross_entropy's batch convention)
    """

    def __init__(self, attention_type: str, hidden_size: int):
        super().__init__()
        self.attention_type = attention_type
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        if self.attention_type == "biaffine":
            ew1 = torch.matmul(self.weight1(encoder_outputs), decoder_output).unsqueeze(1)
            ew2 = self.weight2(encoder_outputs)
            return (ew1 + ew2).permute(1, 0)
        elif self.attention_type == "dot_product":
            return torch.matmul(encoder_outputs, decoder_output).unsqueeze(0)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

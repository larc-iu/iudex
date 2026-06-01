"""Deep biaffine scoring, shared by the biaffine parsers.

Factored out of `topdown_biaffine` when `sr_biaffine` became its second user
(per the "reusable nn.Module" carve-out in CLAUDE.md design choice #1). The
topdown parser scores splits/labels over a span's left/right sub-spans; the
shift-reduce parser scores reduce labels over the top two stack spans. Both
want the same deep biaffine, so it lives here.
"""

import torch.nn as nn


class FeedForward(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )


class DeepBiAffine(nn.Module):
    """Deep biaffine scorer.

    Each side is projected with its own FFN, then combined as a bilinear term
    plus per-side linear terms (a.k.a. the deep biaffine of Dozat & Manning).

    Args:
        h_left: [num_candidates, input_dim]
        h_right: [num_candidates, input_dim]

    Returns:
        scores: [num_candidates, output_dim]
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p, bias=True):
        super().__init__()
        self.W_left = FeedForward(input_dim, hidden_dim, hidden_dim, dropout_p)
        self.W_right = FeedForward(input_dim, hidden_dim, hidden_dim, dropout_p)
        # `bias` toggles a bias term on the bilinear scorer.
        self.W_s = nn.Bilinear(hidden_dim, hidden_dim, output_dim, bias=bias)
        self.V_left = nn.Linear(hidden_dim, output_dim)
        self.V_right = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_left, h_right):
        h_left = self.W_left(h_left)
        h_right = self.W_right(h_right)
        return self.W_s(h_left, h_right) + self.V_left(h_left) + self.V_right(h_right)

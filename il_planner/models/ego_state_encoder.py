"""Present-state ego encoder, used when use_ego_history=False.

Embeds each scalar of the ego state as its own token and pools them with a learned query,
so dropping a channel masks one key rather than feeding a zero the model could read as a
real measurement.

Feeding ego's full history makes extrapolating its own motion the cheapest way to cut
open-loop L2. That inverts in closed loop, where the history is the model's own drift.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# current_state channels, in the order simple_feature._get_ego_current_state writes them.
# The first three are ZEROED by normalize() - ego is the origin of its own frame - so they
# are constants, and dropping them would regularise nothing. That is exactly why they are
# the ones held visible below.
ALWAYS_VISIBLE_CHANNELS = 3


class StateAttentionEncoder(nn.Module):
    """
    Embeds each scalar of the present ego state as its own token, then pools them with a
    learned query via attention - so dropping a channel is just masking one key, and the
    model never sees a zero it could mistake for a real measurement.
    """

    def __init__(self, state_channel: int = 6, dim: int = 128, state_dropout: float = 0.75,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.state_channel = state_channel
        self.state_dropout = state_dropout

        # One Linear PER CHANNEL: a shared projection would make the channels
        # interchangeable, and velocity and steering angle are not interchangeable.
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, state_channel, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        :param current_state: (B, >=state_channel) - extra trailing channels are ignored,
            so the feature can carry yaw_rate without the model being obliged to use it.
        :return: (B, dim) ego token.
        """
        x = current_state[:, : self.state_channel]
        tokens = torch.stack(
            [linear(x[:, i, None]) for i, linear in enumerate(self.linears)], dim=1
        )                                                        # (B, state_channel, dim)
        tokens = tokens + self.pos_embed

        key_padding_mask = None
        if self.training and self.state_dropout > 0:
            # True = ignore this key. The first ALWAYS_VISIBLE_CHANNELS are the zeroed
            # position/heading, kept so a sample can never have every key masked - that
            # would make softmax over all -inf produce NaN.
            visible = torch.zeros(
                (x.shape[0], ALWAYS_VISIBLE_CHANNELS), device=x.device, dtype=torch.bool
            )
            dropped = (
                torch.rand(
                    (x.shape[0], self.state_channel - ALWAYS_VISIBLE_CHANNELS), device=x.device
                )
                < self.state_dropout
            )
            key_padding_mask = torch.cat([visible, dropped], dim=1)

        query = self.query.expand(x.shape[0], -1, -1)
        pooled = self.attn(
            query=query, key=tokens, value=tokens, key_padding_mask=key_padding_mask
        )[0]
        return pooled[:, 0]                                                     # (B, dim)

from torch import nn

from .multihead_custom_attention import MultiheadCustomAttention


class AdaLN(nn.Module):
    """Adaptive LayerNorm - signal-modulated linear transformation."""

    def __init__(self, d_model):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )
        # Initialize as 0 (no scale/shift)
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: tensor (B, S, C)
            t: tensor (B, C)

        Returns:
            tensor (B, S, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DummyLayer(nn.Module):
    """Implement adaptive normalization wrappers, pre-/post-norm, pos embed."""

    def __init__(self, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def _adaln(self, x, layer, ada_sgnl):
        if layer is not None and ada_sgnl is not None:
            return layer(x, ada_sgnl)
        return x

    def forward(self):
        pass


class FFWLayer(DummyLayer):
    """Feed-forward layer for Transformers."""

    def __init__(self, d_model, dim_fw=None, dropout=0.1, use_adaln=False,
                 pre_norm=False):
        super().__init__(pre_norm=pre_norm)
        # Initialize MLP and normalization
        dim_fw = 4 * d_model if dim_fw is None else dim_fw
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_fw),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fw, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

        # Initialize those with Xavier
        self._reset_parameters()

        # Initialize adaptive normalization separately
        self.adaln = None
        if use_adaln:
            self.adaln = AdaLN(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ada_sgnl=None):
        """
        Args:
            x: tensor (B, S, C)
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S, C)
        """
        # Normalize if pre-norm
        x = self._norm(x, self.norm, self.pre_norm)
        # Adaptive normalization if applicable
        x = self._adaln(x, self.adaln, ada_sgnl)
        # Main FFW
        x = x + self.ffn(x)
        # Normalize if post-norm
        x = self._norm(x, self.norm, not self.pre_norm)
        return x


class AttentionLayer(DummyLayer):
    """Attention layer, for self-/cross-attention."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe=False, use_adaln=False, is_self=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__(pre_norm=pre_norm)
        self.rotary_pe = rotary_pe
        self.is_self = is_self  # self-attention, different normalization

        # Normalization and attention layers
        self.adaln = None
        if use_adaln:
            self.adaln = AdaLN(d_model)
        self.attention = MultiheadCustomAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = None
        if pre_norm:
            self.norm_kv = self.norm_q if is_self else nn.LayerNorm(d_model)

    def forward(self, seq1, seq2,
                seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                ada_sgnl=None):
        """
        Args:
            seq1: tensor (B, S1, C)
            seq1_pos: (B, S1, C) if not rotary, else (B, S1, C, 2)
            seq1_sem_pos: (B, S1, C), semantic embedding
            seq2: tensor (B, S2, C)
            seq2_key_padding_mask: tensor (B, S2)
            seq2_pos: (B, S2, C) if not rotary, else (B, S2, C, 2)
            seq2_sem_pos: (B, S2, C), semantic embedding
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S, C)
        """
        # Normalize if pre-norm
        q1 = self._norm(seq1, self.norm_q, self.pre_norm)
        if self.is_self:
            k2 = v2 = self._norm(seq2, self.norm_q, self.pre_norm)
        else:
            k2 = v2 = self._norm(seq2, self.norm_kv, self.pre_norm)
        # Add positional embeddings if not rotary - rotary are handled later
        if not self.rotary_pe:
            q1 = self.with_pos_embed(seq1, seq1_pos)
            k2 = self.with_pos_embed(seq2, seq2_pos)
        # Add semantic embeddings, e.g. ids of each token
        q1 = self.with_pos_embed(q1, seq1_sem_pos)
        k2 = self.with_pos_embed(k2, seq2_sem_pos)
        # Adaptive normalization if applicable
        q1 = self._adaln(q1, self.adaln, ada_sgnl)
        k2 = self._adaln(k2, self.adaln if self.is_self else None, ada_sgnl)
        v2 = self._adaln(v2, self.adaln if self.is_self else None, ada_sgnl)
        # Main attention code
        seq1b = self.attention(
            query=q1.transpose(0, 1),
            key=k2.transpose(0, 1),
            value=v2.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=seq2_key_padding_mask,  # (B, S2)
            rotary_pe=(seq1_pos, seq2_pos) if self.rotary_pe else None
        )[0].transpose(0, 1)
        seq1 = seq1 + self.dropout(seq1b)
        # Normalize if post-norm
        seq1 = self._norm(seq1, self.norm_q, not self.pre_norm)
        return seq1


class AttentionModule(nn.Module):
    """Stacking of attention and feed-forward layers."""

    def __init__(self, num_layers, d_model=256, dim_fw=None,
                 dropout=0.1, n_heads=8, pre_norm=False,
                 rotary_pe=False, use_adaln=False, is_self=False):
        super().__init__()
        self.num_layers = num_layers
        self.is_self = is_self
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(AttentionLayer(
                d_model, dropout, n_heads, pre_norm,
                rotary_pe, use_adaln, is_self
            ))
            self.ffw_layers.append(FFWLayer(
                d_model, dim_fw, dropout, use_adaln, pre_norm=False
            ))

    def forward(self, seq1, seq2,
                seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                ada_sgnl=None):
        """
        Args:
            seq1: tensor (B, S1, C)
            seq2: tensor (B, S2, C)
            seq2_key_padding_mask: tensor (B, S2)
            seq1_pos: (B, S1, C) if not rotary, else (B, S1, C, 2)
            seq2_pos: (B, S2, C) if not rotary, else (B, S2, C, 2)
            seq1_sem_pos: (B, S1, C), semantic embedding
            seq2_sem_pos: (B, S2, C), semantic embedding
            ada_sgnl: tensor (B, C)

        Returns:
            tensor (B, S1, C)
        """
        output = []
        for i in range(self.num_layers):
            if self.is_self:
                seq2 = seq1
            seq1 = self.attn_layers[i](
                seq1, seq2,
                seq2_key_padding_mask,
                seq1_pos, seq2_pos,
                seq1_sem_pos, seq2_sem_pos,
                ada_sgnl
            )
            seq1 = self.ffw_layers[i](seq1, ada_sgnl)
            output.append(seq1)
        return output

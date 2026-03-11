
from torch import nn
from torch.nn import functional as F
import einops

from .position_encodings import RotaryPositionEncoding


class MultiheadCustomAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False,
                 device=None, dtype=None):
        super().__init__(
            embed_dim, num_heads, dropout=dropout,
            bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=batch_first,
            device=device, dtype=dtype
        )

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            attn_mask=None,
            rotary_pe=None):
        r"""Compute attention outputs using query, key, and value embeddings.

        query, key, and value are (S, B, F) if not batch_first else (B, S, F)
        key_padding_mask is (B, S)
        output of same format as query
        """
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        # merge key padding and attention masks
        if key_padding_mask is not None:
            bsz, src_len = key_padding_mask.shape
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
            key_padding_mask = key_padding_mask.reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            attn_mask=attn_mask,
            rotary_pe=rotary_pe)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 training=True,
                                 attn_mask=None,
                                 rotary_pe=None):
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if rotary_pe is not None:  # rotary pe ROPE disentangeld
        qp, kvp = rotary_pe
        q_cos, q_sin = qp[..., 0], qp[..., 1]
        k_cos, k_sin = kvp[..., 0], kvp[..., 1]
        q = RotaryPositionEncoding.embed_rotary(q.transpose(0, 1), q_cos, q_sin).transpose(0, 1)
        k = RotaryPositionEncoding.embed_rotary(k.transpose(0, 1), k_cos, k_sin).transpose(0, 1)

    q = einops.rearrange(q, "S B (H D) -> B H S D", H=num_heads, D=head_dim)
    k = einops.rearrange(k, "S B (H D) -> B H S D", H=num_heads, D=head_dim)
    v = einops.rearrange(v, "S B (H D) -> B H S D", H=num_heads, D=head_dim)

    # Compute attention output
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask, dropout_p if training else 0.0, is_causal=False
    )  # B H S D
    attn_output = einops.rearrange(attn_output, "B H S D -> S B (H D)")
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = F.dropout(attn_output, p=dropout_p, training=training)

    return attn_output, None

import math
import torch
from torch import nn, Tensor
from torch.nn import RMSNorm
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from pixie.models._config import RopeScaling
from . import BaseGPTModel, register_model, ModelConfig, PretrainedConfig


class PixieConfig(ModelConfig): ...


type PositionEmbedding = tuple[Tensor, Tensor]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_attention_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_kv_attention_heads = num_kv_attention_heads or num_attention_heads
        assert num_attention_heads % num_kv_attention_heads == 0
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.num_kv_attention_heads = num_kv_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.kv_rep = num_attention_heads // num_kv_attention_heads
        self.wq = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            hidden_size, num_kv_attention_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            hidden_size, num_kv_attention_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):

        def rotate_half(x: Tensor):
            # x: [batch_size, seq_len, num_attention_heads or num_kv_attention_heads, head_dim]
            # x.shape[-1] is head_dim
            # return shape: same as x, but x[..., : head_dim // 2] and x[..., head_dim // 2 :] are swapped with a sign change
            # last dimension order: [-x_{d/2}, -x_{d/2+1}, ..., -x_{d-1}] + [x_0, x_1, ..., x_{d/2-1}]
            return torch.cat(
                (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
            )

        # cos, sin: [seq_len, head_dim]
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # shape: [seq_len, 1, head_dim]
        # apply rotary embedding
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed.type_as(q), k_embed.type_as(k)

    def forward(self, x: Tensor, position_embedding: PositionEmbedding) -> Tensor:
        batch_size, seq_len, _hidden_size = x.shape
        q = self.wq(x)  # [batch_size, seq_len, num_attention_heads * self.head_dim]
        k, v = self.wk(x), self.wv(
            x
        )  # [batch_size, seq_len, num_kv_attention_heads * self.head_dim]
        # Slice the merged tensors
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_attention_heads, self.head_dim)
        # Apply positional encoding
        cos, sin = position_embedding  # each of shape [seq_len, head_dim]
        q, k = self.apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
        # transpose to [batch_size, num_attention_heads or num_kv_attention_heads, seq_len, head_dim]
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        # Repeat kv heads to match q heads: [batch_size, seq_len, num_attention_heads, head_dim]
        k = k.repeat_interleave(self.kv_rep, -3)
        v = v.repeat_interleave(self.kv_rep, -3)
        # Apply scaled dot-product attention
        dropout_p = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True
        )
        assert output.shape == (
            batch_size,
            self.num_attention_heads,
            seq_len,
            self.head_dim,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        output = self.wo(output)  # [batch_size, seq_len, hidden_size]
        return output  # [batch_size, seq_len, hidden_size]


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        act: str = "silu",
        feed_forward_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feed_forward_size is None:
            feed_forward_size = hidden_size * 4  # Default to 4x hidden size
        self.up_proj = nn.Linear(hidden_size, feed_forward_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, feed_forward_size, bias=False)
        self.down_proj = nn.Linear(feed_forward_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = ACT2FN[act]

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_attention_heads: int | None = None,
        feed_forward_size: int | None = None,
        act: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_attention_heads=num_kv_attention_heads,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            act=act,
            feed_forward_size=feed_forward_size,
            dropout=dropout,
        )
        self.input_norm = RMSNorm(hidden_size, eps=1e-5)
        self.attention_norm = RMSNorm(hidden_size, eps=1e-5)

    def forward(self, x: Tensor, position_embedding: PositionEmbedding):
        x2 = x
        # Attention
        x = (
            self.attention(self.input_norm(x), position_embedding=position_embedding)
            + x2
        )
        # Feed Forward
        x = x + self.feed_forward(self.attention_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: PixieConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        # Embedding layer
        vocab_size = config.get_vocab_size()
        self.tok_emb = nn.Embedding(vocab_size, config.hidden_size)
        self.dropout_emb = nn.Dropout(config.dropout)
        # Positional encoding
        cos, sin = self.precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=self.config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)
        self.freqs_sin: Tensor
        self.freqs_cos: Tensor
        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_kv_attention_heads=config.num_kv_attention_heads,
                    feed_forward_size=config.feed_forward_size,
                    act=config.hidden_act,
                    dropout=config.dropout,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)
        # Final linear layer
        self.out = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self._dynamic_tied_weights_keys = ["out.weight", "tok_emb.weight"]
        self.tok_emb.weight = self.out.weight

    def _tie_weights(self) -> None:
        self._dynamic_tied_weights_keys = ["out.weight", "tok_emb.weight"]
        self.out.weight = self.tok_emb.weight

    def precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        rope_base: float,
        rope_scaling: RopeScaling | None = None,
    ):
        # compute thetas: θ_i = 1 / (base^(2i/d)), where i ∈ [0, d // 2)
        inv_freqs = 1.0 / (
            rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )  # [dim // 2]

        if rope_scaling is not None:
            # YaRN rope scaling: https://arxiv.org/pdf/2309.00071
            assert rope_scaling.type == "yarn", "Only 'yarn' rope scaling is supported."
            # Yarn positional embedding scaling
            orig_max, beta_fast, beta_slow = (
                rope_scaling.original_max_position_embeddings,
                rope_scaling.beta_fast,
                rope_scaling.beta_slow,
            )
            s = end / orig_max
            wavelengths = 2 * math.pi * inv_freqs  # [dim // 2]
            r = orig_max / wavelengths  # [dim // 2]
            gamma = torch.where(
                r < beta_slow,
                torch.zeros_like(r),
                torch.where(
                    r > beta_fast,
                    torch.ones_like(r),
                    (r - beta_slow) / (beta_fast - beta_slow),
                ),
            )
            inv_freqs = (1 - gamma) * (inv_freqs / s) + gamma * inv_freqs
            attention_scale = 0.1 * math.log(s) + 1.0
        else:
            attention_scale = 1.0

        # compute m * θ_i for each position m ∈ [0, end)
        t = torch.arange(end, device=inv_freqs.device)  # [end]
        freqs = torch.outer(t, inv_freqs).float()  # [end, dim // 2]
        freqs = freqs * attention_scale
        # compute cos and sin, each of shape [end, dim]
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        return freqs_cos, freqs_sin

    def forward(self, x: Tensor, **args) -> Tensor:
        _batch_size, seq_len = x.shape
        # Embedding and positional encoding
        tok_embeds = self.tok_emb(x)  # [batch_size, seq_len, hidden_size]
        x = self.dropout_emb(tok_embeds)  # [batch_size, seq_len, hidden_size]
        # Transformer layers
        start = 0
        pos = (
            self.freqs_cos[start : start + seq_len],
            self.freqs_sin[start : start + seq_len],
        )
        for layer in self.layers:
            x = layer(x, position_embedding=pos)  # [batch_size, seq_len, hidden_size]
        # Final normalization and linear layer
        x = self.norm(x)  # [batch_size, seq_len, hidden_size]
        logits = self.out(x)  # [batch_size, seq_len, vocab_size]
        return logits


@register_model("pixie", PixieConfig)
class Pixie(BaseGPTModel[PixieConfig]):
    def __init__(self, config: PixieConfig | PretrainedConfig):
        config = PixieConfig.cast(config)
        super().__init__(config)
        self.model = Transformer(config)

    def forward(self, input_ids: Tensor, **args) -> CausalLMOutputWithPast:
        logits = self.model(input_ids)
        self.out.__setitem__("logits", logits)
        return self.out

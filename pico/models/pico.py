import math
import torch
from torch import nn, Tensor
from torch.nn import RMSNorm
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from ._base import (
    RopeScaling,
    BaseCasualLM,
    register_model,
    ModelConfig,
    PretrainedConfig,
    BasePretrainedConfig,
)
from transformers.cache_utils import Cache
from torch.nn.attention.bias import causal_lower_right


class PicoConfig(ModelConfig): ...


type PositionEmbedding = tuple[Tensor, Tensor]


class KVCache:
    def __init__(self, num_layers: int):
        self.layers: list[tuple[Tensor, Tensor] | None] = [None] * num_layers

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # k, v: [batch_size, seq_len, num_kv_attention_heads, head_dim]
        if self.layers[layer_idx] is None:
            self.layers[layer_idx] = (k, v)
        else:
            x = self.layers[layer_idx]
            assert x
            cached_k, cached_v = x
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            self.layers[layer_idx] = (k, v)
        return k, v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_attention_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
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

    def forward(
        self,
        x: Tensor,
        position_embedding: PositionEmbedding,
        past_key_values: KVCache | None = None,
    ) -> Tensor:
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
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        # KV cache
        if past_key_values is not None:
            k, v = past_key_values.update(self.layer_idx, k, v)
        # transpose to [batch_size, num_attention_heads or num_kv_attention_heads, seq_len, head_dim]
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        # Apply scaled dot-product attention
        dropout_p = self.dropout if self.training else 0.0
        if past_key_values is not None:
            attn_mask = causal_lower_right(seq_len, k.shape[2])
            is_causal = False
        else:
            attn_mask = None
            is_causal = True
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=self.kv_rep > 1,
            attn_mask=attn_mask,
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
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_attention_heads: int | None = None,
        feed_forward_size: int | None = None,
        act: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            layer_idx=layer_idx,
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

    def forward(
        self,
        x: Tensor,
        position_embedding: PositionEmbedding,
        past_key_values: KVCache | None = None,
    ) -> Tensor:
        x2 = x
        # Attention
        x = self.attention(
            self.input_norm(x),
            position_embedding=position_embedding,
            past_key_values=past_key_values,
        )
        x = x + x2
        # Feed Forward
        x = x + self.feed_forward(self.attention_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: PicoConfig):
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
                    layer_idx=i,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_kv_attention_heads=config.num_kv_attention_heads,
                    feed_forward_size=config.feed_forward_size,
                    act=config.hidden_act,
                    dropout=config.dropout,
                )
                for i in range(self.num_hidden_layers)
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

    def forward(
        self, x: Tensor, past_key_values: KVCache | None = None, **args
    ) -> Tensor:
        _batch_size, seq_len = x.shape
        # Embedding and positional encoding
        tok_embeds = self.tok_emb(x)  # [batch_size, seq_len, hidden_size]
        x = self.dropout_emb(tok_embeds)  # [batch_size, seq_len, hidden_size]
        # Transformer layers
        if past_key_values is None or past_key_values.layers[0] is None:
            start = 0
        else:
            start = past_key_values.layers[0][0].shape[1]
        pos = (
            self.freqs_cos[start : start + seq_len],
            self.freqs_sin[start : start + seq_len],
        )
        for layer in self.layers:
            x = layer(
                x, position_embedding=pos, past_key_values=past_key_values
            )  # [batch_size, seq_len, hidden_size]
        # Final normalization and linear layer
        x = self.norm(x)  # [batch_size, seq_len, hidden_size]
        logits = self.out(x)  # [batch_size, seq_len, vocab_size]
        return logits


class PicoPretrainedConfig(BasePretrainedConfig):
    model_name: str = "Pico"


@register_model("pico", PicoConfig)
class Pico(BaseCasualLM[PicoConfig]):
    config_class = PicoPretrainedConfig

    def __init__(self, config: PicoConfig | PretrainedConfig):
        config = PicoConfig.cast(config)
        super().__init__(config)
        self.model = Transformer(config)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Cache | KVCache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if use_cache:
            if isinstance(past_key_values, Cache) or past_key_values is None:
                cache = KVCache(self.config.num_hidden_layers)
            else:
                cache = past_key_values
        else:
            cache = None
        logits = self.model(input_ids, past_key_values=cache)
        if "labels" in kwargs:
            loss = self.loss_function(
                logits=logits,
                # labels=kwargs["labels"],
                vocab_size=self.args.get_vocab_size(),
                **kwargs,
            )
        else:
            loss = None
        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            past_key_values=cache,  # type: ignore
        )

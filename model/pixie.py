import torch
from torch import nn, Tensor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from torch.nn import RMSNorm
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.config import Config


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

    def repeat_kv(self, x: Tensor) -> Tensor:
        batch_size, seq_len, num_kv_attention_heads, head_dim = x.shape
        if self.kv_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, num_kv_attention_heads, self.kv_rep, head_dim)
            .reshape(
                batch_size, seq_len, num_kv_attention_heads * self.kv_rep, head_dim
            )
        )

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        def rotate_half(x):
            return torch.cat(
                (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
            )

        q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
            rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
        )
        k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
            rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
        )
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
        cos, sin = position_embedding
        q, k = self.apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
        # Repeat kv heads to match q heads: [batch_size, seq_len, num_attention_heads, head_dim]
        k, v = self.repeat_kv(k), self.repeat_kv(v)
        # transpose to [batch_size, num_attention_heads, seq_len, head_dim]
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
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
    def __init__(self, config: Config | None = None):
        super().__init__()
        config = config or Config()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        # Embedding layer
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout_emb = nn.Dropout(config.dropout)
        # Positional encoding
        cos, sin = self.precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=32768,
            theta=config.rope_theta,
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
        self.out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tok_emb.weight = self.out.weight

    def precompute_freqs_cis(self, dim: int, end: int, theta: float):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
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
        logits = self.out(x)  # [batch_size, seq_len, hidden_size]
        return logits


class Pixie(PreTrainedModel, GenerationMixin):
    NAME: str = "pixie"

    def __init__(self, config: Config | None = None, compile: bool = True):
        self.config = config or Config()
        super().__init__(self.config)
        self.model = Transformer(self.config)
        self.out = CausalLMOutputWithPast()
        if compile:
            self.model = torch.compile(self.model, mode="default")

    def forward(self, input_ids: Tensor, **args) -> CausalLMOutputWithPast:
        logits = self.model(input_ids)
        self.out.__setitem__("logits", logits)
        return self.out

    @staticmethod
    def tokenizer(config: Config) -> PreTrainedTokenizerFast:
        """
        Creates the tokenizer for the model.
        """
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer


if __name__ == "__main__":
    # Print important stats about the model
    config = Config()
    model = Pixie(config, compile=False)
    tokenizer = Pixie.tokenizer(config)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.NAME}")
    print(f"Total parameters: {parameters / 1e6:.3f} M")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Tokenizer: {config.tokenizer}")
    print(f"Context length: {config.context_length}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention Layers: {config.num_hidden_layers}")
    print(
        f"Attention heads: Q={config.num_attention_heads}, KV={config.num_kv_attention_heads}"
    )
    print(f"Feed forward size: {config.feed_forward_size}")

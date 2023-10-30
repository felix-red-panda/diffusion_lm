import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion:
    def __init__(self, noise_steps=1200, beta_start=1e-4, beta_end=0.02) -> None:
        pass

    def prepare_noise_schedule(self):
        pass

    def noise_text(self):
        pass

    def sample_timestep(self):
        pass

    def sample(self):
        pass


# Layer normalization with optional bias
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# Self Attention Head without flash attention
class SelfAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        # get query and key projections
        q = self.query(x)  # (B, T, C)
        k = self.key(x)  # (B, T, C)
        # compute attention "affinities", scale, mask, and softmax
        att = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        att = att * C ** (-0.5)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        # apply attention to value projection
        v = self.value(x)  # (B, T, C)
        out = att @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


# Cross Attention Head without flash attention
class CrossAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, e, x):
        # e: from encoder, x: from decoder
        B, T, C = x.shape
        # get query and key projections
        q = self.query(x)  # (B, T, C)
        k = self.key(e)  # (B, T, C)
        # compute attention "affinities", scale, mask, and softmax
        att = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        att = att * C ** (-0.5)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        # apply attention to value projection
        v = self.value(e)  # (B, T, C)
        out = att @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


# Parallel self attention heads
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )  # linear for dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # compute and concat heads in parallel
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # project and dropout
        out = self.dropout(self.proj(out))
        return out


# Parallel cross attention heads
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [CrossAttentionHead(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )  # linear for dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, e, x):
        # compute and concat heads in parallel
        out = torch.cat([head(e, x) for head in self.heads], dim=-1)
        # project and dropout
        out = self.dropout(self.proj(out))
        return out


# Feedforward layer
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # linear, gelu, linear, dropout
        out = self.lin1(x)
        out = self.gelu(out)
        out = self.lin2(out)
        out = self.dropout(out)
        return out


# Encoder block, full self attention on language utterance
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.ln2 = LayerNorm(config.n_embd, config.bias)

    def forward(self, x):
        # layer norm, attention, residual
        out = x + self.sa(self.ln1(x))
        # layer norm, feedforward, residual
        e = out + self.ff(self.ln2(out))
        return e


# Denoiser block, takes noisy input and encoded utterance, and predicts denoised embedding
class DenoiserBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ca = MultiHeadCrossAttention(config)
        self.sa = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.ln3 = LayerNorm(config.n_embd, config.bias)

    def forward(self, e, x):
        # e: encoded utterance, x: noisy input
        # layer norm, cross attention, residual
        out = x + self.ca(self.ln1(e), self.ln1(x))
        # layer norm, self attention, residual
        out = out + self.sa(self.ln2(out), self.ln2(out))
        # layer norm, feedforward, residual
        x0 = out + self.ff(self.ln2(out))
        return x0


# Decoder block, takes denoised embedding and encoded utterance, and predicts output embedding (to be projected to tokens)
# same as denoiser block, just copied to match with paper for clarity
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ca = MultiHeadCrossAttention(config)
        self.sa = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.ln2 = LayerNorm(config.n_embd, config.bias)

    def forward(self, e, x0):
        # e: encoded utterance, x0: denoised embedding
        # layer norm, cross attention, residual
        out = x0 + self.ca(self.ln1(e, x0), self.ln1(x0))
        # layer norm, self attention, residual
        out = out + self.sa(self.ln1(out), self.ln1(out))
        # layer norm, feedforward, residual
        d = out + self.ff(self.ln2(out))
        return d


# Completely based on the text in 3.1 Architecture of CodeFusion paper
class CodeFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)
        # no idea if we should add positional embedding here
        self.pos_embd = nn.Embedding(config.ctx_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.encoders = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.n_layer)]
        )
        self.denoisers = nn.ModuleList(
            [DenoiserBlock(config) for _ in range(config.n_layer)]
        )
        self.decoders = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layer)]
        )
        self.ln = LayerNorm(config.n_embd, config.bias)
        # "classification head" H
        self.h = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, n, targets=None):
        # x: tokenized utterance, n: noisy input, targets: tokenized output
        b, t = x.size()
        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        # shape (1, t)
        # get token and position embeddings
        tok_embd = self.tok_embd(x)  # (B, T, C)
        # again, no idea if we should add positional embedding here
        pos_embd = self.pos_embd(pos)  # (1, T, C)
        # add them up, apply dropout
        x = self.dropout(tok_embd + pos_embd)  # (B, T, C)
        # apply encoders to get encoded utterance
        e = self.encoders(x)  # (B, T, C)
        # apply denoisers to get denoised embedding
        x0 = self.denoisers(e, n)
        # apply decoders to get output embedding
        d = self.decoders(e, x0)

        if targets is not None:
            # if we are training
            logits = self.h(x)  # (B, T, V)
            # TODO: implement the proper loss for diffusion
            # loss =
        else:
            # if we are just doing inference
            logits = self.h(d)  # (B, T, V)
            loss = None
        return logits, loss

    # Decoder forward pass for unsupervised training phase
    def decoder_forward(self, n, targets):
        # n: noisy input, targets: tokenized output
        # for unsupervised training, e is just gaussian noise (see 3.2 Training (?))
        e = torch.randn_like(n)
        x0 = self.denoisers(e, n)
        d = self.decoders(e, x0)

        logits = self.h(d)  # (B, T, V)
        # idk if this is the right loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

        return logits, loss

    @torch.no_grad()
    def generate(self, x, time_steps, temperature=1.0):
        for _ in range(time_steps):
            # TODO: implement
            pass
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

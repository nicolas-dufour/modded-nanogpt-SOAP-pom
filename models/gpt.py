import torch
import torch.nn as nn
import torch.nn.functional as F
import models.pom as pom


def rmsnorm(x0, eps=1e-6):
    """RMS normalization function (matching reference implementation)."""
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

def justnorm(x0):
    x = x0.float()
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    return x.type_as(x0)

class RMSNorm(nn.Module):
    """RMS normalization module."""
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.type_as(x0)


class Norm(nn.Module):
    """Normalization module to reproject on the unit sphere."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(p=2, dim=-1, keepdim=True)

class Rotary(torch.nn.Module):
    """Rotary position embeddings."""
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfPoM(nn.Module):
    """Causal self-attention using Polynomial Mixer."""
    
    def __init__(self, n_embd, degree, expand, n_head):
        super().__init__()
        self.degree = degree
        self.expand = expand
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        self.pom = pom.PoM(self.n_embd, self.degree, self.expand, False)

    def forward(self, x: torch.Tensor, sqk: dict) -> torch.Tensor:
        B, T, C = x.size()
        mask = torch.tril(torch.ones((T, T))).unsqueeze(0)
        return self.pom(x, x, mask, sqk)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, degree, expand, n_head):
        super().__init__()
        self.degree = degree
        self.expand = expand
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """Multi-layer perceptron block."""
    
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Transformer block with PoM attention and MLP."""
    
    def __init__(self, mixing_layer, n_embd, n_layer, use_nGPT):
        super().__init__()
        self.attn = mixing_layer #CausalSelfPoM(n_embd, degree, expand, n_head)
        self.mlp = MLP(n_embd)
        self.attn_scale = (1 / (2 * n_layer)**0.5)
        self.use_nGPT = use_nGPT
        self.sqk = None

        if self.use_nGPT == 1:
            # For attn
            self.attn_alpha_init_value = 0.5
            self.attn_alpha_init_scaling = 1 / (n_embd**0.5)
            self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(n_embd, dtype=torch.float32))

            # For mlp
            self.mlp_alpha_init_value = 0.5
            self.mlp_alpha_init_scaling = 1 / (n_embd**0.5)
            self.mlp_alpha = nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(n_embd, dtype=torch.float32))

            # For q and k: q -> sigmoid(Ws*X), k -> H(X)
            k = self.attn.degree
            dim = self.attn.n_embd
            expand = self.attn.expand
            self.sqk = {"init_value": 1.0, 
                        "init_scaling": 1 / (n_embd**0.5)}
            self.sqk["sqk"] = nn.Parameter(self.sqk["init_scaling"]*torch.ones(k*dim*expand, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_nGPT == 0:
            x = x + self.attn_scale * self.attn(rmsnorm(x))
            x = x + self.mlp(rmsnorm(x))

        if self.use_nGPT == 1:
            x_norm = justnorm(x)

            # For attn
            lr_attn = self.attn_alpha * (self.attn_alpha_init_value * self.attn_alpha_init_scaling)
            lr_attn = torch.abs(lr_attn)
            x_attn = x_norm + lr_attn * (justnorm(self.attn(x_norm, self.sqk)) - x_norm)
            x_attn = justnorm(x_attn)

            # For mlp
            lr_mlp = self.mlp_alpha * (self.mlp_alpha_init_value * self.mlp_alpha_init_scaling)
            lr_mlp = torch.abs(lr_mlp)
            x_mlp = x_attn + lr_mlp * (justnorm(self.mlp(x_attn)) - x_attn)
            x_mlp = justnorm(x_mlp)

        return x_mlp


class GPT(nn.Module):
    """GPT model with Polynomial Mixer attention."""
    
    def __init__(self, mixing_layer, vocab_size: int = 50257, n_layer: int = 12, n_head: int = 12, n_embd: int = 768, use_nGPT: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.use_nGPT = use_nGPT
        self.head_dim = self.n_embd // self.n_head

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.n_embd),
            h=nn.ModuleList([Block(mixing_layer, self.n_embd, self.n_layer, self.use_nGPT) for _ in range(self.n_layer)]),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.rotary = Rotary(self.head_dim)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_logits: bool = True):
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices for loss computation
            return_logits: Whether to return logits
            
        Returns:
            Tuple of (logits, loss) if targets provided, else just logits
        """
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(x)
        x = apply_rotary_emb(x, cos, sin)
        x = x.view(B, T, C)

        for block in self.transformer.h:
            x = block(x)

        if self.use_nGPT == 0:
            x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple):
        """
        Configure optimizers for the model.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam betas
            
        Returns:
            Combined optimizer
        """
        from models.optimizers.combined_optimizer import CombinedOptimizer
        from models.optimizers.adamw_optimizer import AdamWOptimizer
        from models.optimizers.soap import SOAP  # Import raw SOAP class
        
        # Create optimizers for different parameter groups
        optimizers = []
        
        # AdamW for lm_head parameters (wte weights are tied to lm_head, so we only include lm_head)
        lm_head_optimizer = AdamWOptimizer(
            self.lm_head.parameters(),
            lr=0.0018,  # Fixed learning rate for lm_head
            betas=betas,
            weight_decay=0  # No weight decay for lm_head
        )
        optimizers.append(lm_head_optimizer)
        
        # SOAP for transformer layers - use raw SOAP class like reference
        transformer_optimizer = SOAP(
            self.transformer.h.parameters(),
            lr=learning_rate,
            betas=(0.95, 0.95),  # Fixed betas for SOAP
            weight_decay=0,  # No weight decay for transformer layers
            precondition_frequency=10  # Fixed precondition frequency
        )
        optimizers.append(transformer_optimizer)
        
        return CombinedOptimizer(optimizers) 
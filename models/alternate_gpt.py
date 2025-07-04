import torch.nn as nn
from models.gpt import GPT, Block, CausalSelfAttention, apply_rotary_emb
import torch.nn.functional as F
import torch



# taken from https://github.com/meta-llama/llama-models/blob/main/models/llama4/model.py#L431
def create_chunked_attention_mask(seq_len: int, attention_chunk_size: int, device: torch.device) -> torch.Tensor:
    block_pos = torch.abs(
        (torch.arange(seq_len).unsqueeze(0) // attention_chunk_size)
        - (torch.arange(seq_len).unsqueeze(1) // attention_chunk_size)
    )
    token_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    mask = (block_pos == 0) & (token_pos <= 0)
    return mask.to(device)


class CausalLocalSelfAttention(CausalSelfAttention):

    def __init__(self, n_embd, degree, expand, n_head, attention_chunk_size=128):
        super().__init__(n_embd, degree, expand, n_head)
        self.attention_chunk_size = attention_chunk_size

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        attention_chunk_size = self.attention_chunk_size
        local_attn_mask = create_chunked_attention_mask(T, attention_chunk_size, x.device)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=local_attn_mask, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class AlternateGPT(GPT):
    """Hybrid GPT model with alternating PoM and Self-Attention layers."""
    
    def __init__(self, mixing_layer, vocab_size: int = 50257, n_layer: int = 12, n_head: int = 12, n_embd: int = 768, 
                 degree: int = 2, expand: int = 2, mixing_layer_alternate: nn.Module = None, pom_to_sa_ratio: float = 5):
        """
        Args:
            pom_to_sa_ratio: Ratio of PoM to self-attention layers (default 5 = 1 sa layer to 5 pom layers)
        """
        # if n_layer % (pom_to_sa_ratio + 1) != 0:
        #     raise ValueError(f"n_layer must be a multiple of (pom_to_sa_ratio + 1) for proper hybrid ratio, got {n_layer}")
        
        if mixing_layer_alternate is None:
            mixing_layer_alternate = CausalSelfAttention(n_embd, degree, expand, n_head)
        
        mixing_layers = []
        for i in range(n_layer):
            if i % (pom_to_sa_ratio + 1) == pom_to_sa_ratio:  # Every pom_to_sa_ratio + 1 layers
                mixing_layers.append(mixing_layer_alternate)
            else:
                mixing_layers.append(mixing_layer)
        
        super().__init__(mixing_layer=None, vocab_size=vocab_size, n_layer=n_layer, 
                        n_head=n_head, n_embd=n_embd)
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.n_embd),
            h=nn.ModuleList([Block(mixing_layer, self.n_embd, self.n_layer) for mixing_layer in mixing_layers]),
        ))
        
        self.transformer.wte.weight = self.lm_head.weight
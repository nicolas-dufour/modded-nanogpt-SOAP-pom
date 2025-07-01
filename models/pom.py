import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from typing import Optional, Tuple, Dict, Any

torch._dynamo.config.suppress_errors = True

# =============================================================================
# Core Polynomial Functions
# =============================================================================

def justnorm(x0):
    x = x0.float()
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    return x.type_as(x0)

@torch.compile
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation function with torch.compile optimization."""
    return F.gelu(x)

@torch.compile
def po2(x: torch.Tensor) -> torch.Tensor:
    """
    Second-order polynomial expansion.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor of shape (..., 2*dim) with polynomial interactions
    """
    h1, h2 = gelu(x).chunk(2, dim=-1)
    h2 = h2 * h1
    return torch.cat([h1, h2], dim=-1)

@torch.compile
def po3(x: torch.Tensor) -> torch.Tensor:
    """
    Third-order polynomial expansion.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor of shape (..., 3*dim) with polynomial interactions
    """
    h1, h2, h3 = gelu(x).chunk(3, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    return torch.cat([h1, h2, h3], dim=-1)

@torch.compile
def po4(x: torch.Tensor) -> torch.Tensor:
    """
    Fourth-order polynomial expansion.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor of shape (..., 4*dim) with polynomial interactions
    """
    h1, h2, h3, h4 = gelu(x).chunk(4, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    h4 = h4 * h3
    return torch.cat([h1, h2, h3, h4], dim=-1)

# =============================================================================
# Masking and Aggregation Functions
# =============================================================================

def mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D mask mixing for attention.
    
    Args:
        h: Hidden states tensor of shape (batch, seq_len, dim)
        mask: Attention mask of shape (batch, seq_len)
        
    Returns:
        Masked and aggregated tensor of shape (batch, 1, dim)
    """
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True) / (1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))

def full_mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply 3D mask mixing for cross-attention.
    
    Args:
        h: Hidden states tensor of shape (batch, seq_len, dim)
        mask: Attention mask of shape (batch, query_len, seq_len)
        
    Returns:
        Masked and aggregated tensor of shape (batch, query_len, dim)
    """
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h

# =============================================================================
# Polynomial Aggregation and Selection
# =============================================================================

def polynomial_aggregation_(x: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply polynomial aggregation with optional masking.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        k: Polynomial order (2, 3, 4, or higher)
        mask: Optional attention mask
        
    Returns:
        Aggregated tensor with polynomial interactions
    """
    # Use optimized functions for common cases
    if k == 2:
        h = po2(x)
    elif k == 3:
        h = po3(x)
    elif k == 4:
        h = po4(x)
    else:
        # Generic case for k > 4
        h = list(gelu(x).chunk(k, dim=-1))
        for i in range(1, k):
            h[i] = h[i] * h[i-1]
        h = torch.cat(h, dim=-1)
    
    # Apply masking if provided
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim() == 2:
            h = mask_mixer(h, mask.to(h.device))
        elif mask.dim() == 3:
            h = full_mask_mixer(h, mask.to(h.device))
        else:
            raise ValueError(f'Unsupported mask dimension: {mask.dim()}. Expected 2, 3, or None.')
    return h

@torch.compile
def polynomial_selection_(x: torch.Tensor, h: torch.Tensor, sqk: dict = None, k: int = None, dim: int = None, expand: int = None) -> torch.Tensor:
    """
    Apply polynomial selection with sigmoid gating.
    
    Args:
        x: Query tensor
        h: Context tensor from polynomial aggregation
        
    Returns:
        Gated output tensor
    """
    if sqk is not None:
        scale_qk = (sqk["sqk"] * (sqk["init_value"]/sqk["init_scaling"])).view(1, 1, k * dim * expand).to(x.device)
        x = scale_qk * justnorm(F.sigmoid(x))  
        h = scale_qk * justnorm(h)
    return x * h

# =============================================================================
# Main PoM Function
# =============================================================================

def pom(xq: torch.Tensor, xc: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None, sqk: dict = None, dim: int = None, expand: int = None) -> torch.Tensor:
    """
    Polynomial Mixer (PoM) operation.
    
    This function implements the polynomial mixer operation which combines
    polynomial aggregation of context with selection from queries.
    
    Args:
        xq: Query input tensor of shape (batch, query_len, dim)
        xc: Context input tensor of shape (batch, context_len, dim)
        k: Polynomial order (degree of interactions to capture)
        mask: Optional attention mask for masking specific positions
        
    Returns:
        Output tensor after polynomial mixing
    """
    h = polynomial_aggregation_(xc, k, mask)
    o = polynomial_selection_(xq, h, sqk, k, dim, expand)

    return o

# =============================================================================
# PoM Module Class
# =============================================================================

class PoM(nn.Module):
    """
    Polynomial Mixer (PoM) Module.
    
    A custom neural network layer designed for capturing higher-order interactions 
    between input features through polynomial expansions. This module consists of
    three linear projections and a custom PoM operation.
    
    Attributes:
        dim (int): The dimensionality of the input features
        order (int): The order of the polynomial interactions to capture
        order_expand (int): The expansion factor for the polynomial order
        po_proj (nn.Linear): Linear projection for polynomial computation
        se_proj (nn.Linear): Linear projection for selection mechanism
        ag_proj (nn.Linear): Linear projection for output aggregation
        pom (callable): The polynomial mixer operation function
    """
    
    def __init__(self, dim: int, degree: int, expand: int, bias: bool = True):
        """
        Initialize the PoM module.
        
        Args:
            dim: The dimensionality of the input features
            degree: The degree of the polynomial to capture
            expand: The expansion factor for the polynomial order
            bias: Whether to include bias terms in linear projections
        """
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand

        # Linear projections
        self.po_proj = nn.Linear(dim, degree * expand * dim, bias=bias)
        self.se_proj = nn.Linear(dim, degree * expand * dim, bias=bias)
        self.ag_proj = nn.Linear(degree * expand * dim, dim, bias=bias)
        self.pom = pom

    def forward(self, xq: torch.Tensor, xc: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, sqk: dict = None) -> torch.Tensor:
        """
        Forward pass of the PoM module.
        
        Args:
            xq: Query input tensor of shape (batch, n_tokens, dim)
            xc: Context input tensor. If None, self-attention is performed
            mask: Optional attention mask tensor
            
        Returns:
            Output tensor after applying the PoM operation
        """
        if xc is None:
            xc = xq  # self-attention

        s = self.se_proj(xq)
        h = self.po_proj(xc)
        sh = self.pom(s, h, self.order, mask, sqk, self.dim, self.order_expand)

        return self.ag_proj(sh)

    def state_forward(self, xq: torch.Tensor, xc: Optional[torch.Tensor] = None, 
                     state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with state management for incremental processing.
        
        Args:
            xq: Query input tensor
            xc: Context input tensor. If None, self-attention is performed
            state: Optional state dictionary from previous forward pass
            
        Returns:
            Tuple of (output_tensor, new_state)
        """
        if xc is None:
            xc = xq  # self-attention

        s = self.se_proj(xq)
        xc = self.po_proj(xc)
        h_current = polynomial_aggregation_(xc, self.order)
        n_current = h_current.shape[1]

        if state is not None:
            h_past = state['h']
            n_past = state['n']
            h = (n_past * h_past + n_current * h_current) / (n_past + n_current)
        else:
            h = h_current
            n_past = 0

        new_state = {'h': h, 'n': n_past + n_current}

        sh = polynomial_selection_(s, h)
        return self.ag_proj(sh), new_state
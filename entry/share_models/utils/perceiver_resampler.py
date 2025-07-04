import torch
from torch import nn, einsum
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding


from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many


def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# Define a standard transformer block for latents
class LatentTransformerBlock(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64, ff_mult=2, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Using batch_first=True to work with shape [batch, seq, dim]
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim)
        )
        
    def forward(self, x):
        # x: (batch, num_latents, dim)
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        return x


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        use_rotary=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.use_rotary = use_rotary
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(
                dim = int(dim_head * 0.5),
            )

    def forward(self, x, latents, attention_mask=None):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((latents, x), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        if self.use_rotary: 
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
                    
        q = q * self.scale
        
        
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        
        # Apply key padding mask if provided
        if attention_mask is not None:
            # attention_mask: (b, key_len)
            # Expand the mask for heads and queries
            query_attn_mask = torch.zeros(latents.shape[:2]).to(attention_mask.device)
            attention_mask = torch.cat([query_attn_mask, attention_mask], dim=1)
            attention_mask = repeat(attention_mask, 'b j -> b h i j', h=sim.shape[1], i=sim.shape[2])
            sim = sim.masked_fill(attention_mask == 1, float('-inf'))


        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 4,
        max_seq_len = 16,
        ff_mult = 2,
        use_rotary = False,
        use_latent_transformer=False,
    ):
        super().__init__()
        self.use_rotary = use_rotary
        self.max_seq_len = max_seq_len
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        if not use_rotary: self.media_pos_emb = nn.Parameter(torch.randn(max_seq_len, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads, use_rotary=use_rotary),
                LatentTransformerBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult) if use_latent_transformer else nn.Identity()
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, attention_mask=None):

        seq_len = x.shape[1]
        
        if not self.use_rotary:
            pos_embedding = repeat(self.media_pos_emb, 'n d -> b n d', b = x.shape[0])

            if seq_len <= self.max_seq_len:
                pos_embedding = pos_embedding[:, :seq_len, :]  # Match seq_len (1, seq_len, latent_dim)
            else:
                pos_embedding = F.interpolate(
                    rearrange(pos_embedding, 'b t d -> b d t'), 
                    size=seq_len,            
                    mode='linear',
                    align_corners=False           
                )
                pos_embedding = rearrange(pos_embedding, 'b d t -> b t d')
            
            x = x + pos_embedding[:, :seq_len]

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        for attn, lformer in self.layers:
            latents = attn(x, latents, attention_mask) + latents
            latents = lformer(latents)
            # latents = ff(latents) + latents
        norm_latents = self.norm(latents)
        return norm_latents
        
    
    
    

    
    


class PerceiverResamplerv0(nn.Module):
    def __init__(self, latent_dim, num_latents, input_dim, num_heads=8, max_seq_len=16):
        super(PerceiverResamplerv0, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))  # learnable latent array
        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads)
        self.input_to_latent_proj = nn.Linear(input_dim, latent_dim)  # project input to latent_dim
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))  # (1, max_seq_len, latent_dim)
        self.max_seq_len = max_seq_len
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, input_vectors, attention_mask=None):
        """
        input_vectors: (batch_size, seq_len, input_dim)
        attention_mask: (batch_size, seq_len) where 1 indicates valid and 0 indicates padding
        
        """
        
        batch_size, seq_len, input_dim = input_vectors.shape
        latent_queries = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_latents, latent_dim)
        
        # Project input to latent_dim
        projected_inputs = self.input_to_latent_proj(input_vectors)  # (batch_size, seq_len, latent_dim)
        
        if seq_len <= self.max_seq_len:
            pos_embedding = self.positional_embedding[:, :seq_len, :]  # Match seq_len (1, seq_len, latent_dim)
        else:
            pos_embedding = F.interpolate(
                rearrange(self.positional_embedding, 'b t d -> b d t'), 
                size=seq_len,            
                mode='linear',
                align_corners=False           
            )
            pos_embedding = rearrange(pos_embedding, 'b d t -> b t d')
        projected_inputs = projected_inputs + pos_embedding  # Add positional embedding

        
        # Transpose for compatibility with nn.MultiheadAttention: (seq_len, batch_size, latent_dim)
        projected_inputs = projected_inputs.permute(1, 0, 2)
        latent_queries = latent_queries.permute(1, 0, 2)
        
        # Use cross-attention to map input to latent queries
        resampled_latents, _ = self.cross_attention(query=latent_queries, key=projected_inputs, value=projected_inputs, 
                                                    key_padding_mask=attention_mask)
        
        # Transpose back: (batch_size, num_latents, latent_dim)
        resampled_latents = resampled_latents.permute(1, 0, 2)
        resampled_latents = self.layer_norm(resampled_latents)
        
        return resampled_latents



if __name__ == '__main__':
    def count_all_parameters(model):
        return sum(p.numel() for p in model.parameters())

    perceiver = PerceiverResampler(dim=1024, depth=1, num_latents=1, use_rotary=True)
    perceiverv0 = PerceiverResamplerv0(latent_dim=1024, num_latents=1, input_dim=64)
    
    print(count_all_parameters(perceiver))
    print(count_all_parameters(perceiverv0))
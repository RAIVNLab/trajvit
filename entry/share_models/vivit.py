import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)

class Vivit(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        model_name="vit-base",
        pool = 'cls',
        dropout = 0.,
        emb_dropout = 0.,
        variant = 'factorized_encoder',
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        # self.num_frames//self.tubelet_size, self.patch_num, self.embed_dim
        
        dim_head=64
        channels=3
        if model_name == 'vit-base':
            dim=768
            spatial_depth=6          # half of total 12
            temporal_depth=6         # half of total 12
            self.num_heads=heads=12
            mlp_dim=3072
        elif model_name == 'vit-large':
            dim=1024
            spatial_depth=12       # half of total 24
            temporal_depth=12      # half of total 24
            self.num_heads=heads=16
            mlp_dim=4096
        else:
            raise NotImplementedError
        

        self.num_frames = frames
        self.tubelet_size = frame_patch_size
        self.patch_num = (image_size // image_patch_size) ** 2
        self.embed_dim = dim
        
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.variant = variant


    def interpolate_pos_embedding(self, new_t):
        pos_token = self.pos_embedding.reshape(
            -1, self.num_frames//self.tubelet_size, self.patch_num, self.embed_dim)
        pos_token = rearrange(pos_token, 'b t p d -> b (p d) t')
        pos_token = torch.nn.functional.interpolate(
            pos_token, size=new_t, mode='linear', align_corners=False)
        pos_token = rearrange(pos_token, 'b (p d) t -> b t p d', p=self.patch_num, d=self.embed_dim)
        # pos_token = rearrange(pos_token, 'b t p d -> b (t p) d')
        return pos_token

    def forward(self, video):
        if video.shape[2] == 3 : # default t channel in dim 2
            video = rearrange(video, "b t c w h -> b c t w h")
        else:
            assert video.shape[1] == 3
        
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        if x.size(1) <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :f, :n]
        else:
            x = x + self.interpolate_pos_embedding(x.size(1))

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')

            # attend across space

            x = self.spatial_transformer(x)
            x = rearrange(x, '(b f) n d -> b f n d', b = b)

            # excise out the spatial cls tokens or average pool for temporal attention

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

            # append temporal CLS tokens

            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

                x = torch.cat((temporal_cls_tokens, x), dim = 1)
            

            # attend across time

            x = self.temporal_transformer(x)

            # excise out temporal cls token or average pool

            # x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'factorized_self_attention':
            x = self.factorized_transformer(x)
            # x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')
            
        return x
    
    
    
if __name__ == "__main__":
    v = Vivit(
    image_size=224,           # could be 128, 224, etc.
    frames=16,
    image_patch_size=16,
    frame_patch_size=2,
    model_name='vit-large'
    ).cuda()
    
    for t in [64,128,256,512]:
        video = torch.zeros(1,3,t,224,224).cuda()
        print(t, "--", v(video).shape)
        
        # 256
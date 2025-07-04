import torch
from torch import nn
from easydict import EasyDict as edict
import math
from einops import rearrange, repeat
import logging

from share_models.utils.tensors import compute_centers, decompose_masks, compute_bounding_boxes, decompose_masks_no_t
from share_models.utils.perceiver_resampler import PerceiverResampler, PerceiverResamplerv0
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SinuPosEncoding():
    """
    Computes the positional encoding for n-dimensional coordinates in a fully vectorized manner for batched input.

    coords: Tensor of shape (batch_size, d) representing batched d-dimensional coordinates
    d_output: Desired output dimensionality of the positional encoding
    """
    def __init__(self, d_coords, d_output):
        self.d_coords = d_coords
        self.d_output = d_output
        assert self.d_output % self.d_coords == 0  # Ensure even division
    
    def __call__(self, coords):
        # Expect coords to have shape (batch_size, n_coords, d_coords)
        b,n,d = coords.shape
        coords = rearrange(coords, 'b n d -> (b n) d')
        batch_size, d = coords.shape
        assert d == self.d_coords
        
        d_model_per_coord = self.d_output // d
        extra_dims = self.d_output % d
        
        # Indices and div_term, same for all batches
        indices = torch.arange(self.d_output, dtype=torch.float32, device=coords.device)
        div_term = torch.exp(-math.log(10000.0) * indices / self.d_output)
        
        # Expand coordinates to match the desired output shape
        coords_expanded = coords.unsqueeze(2).expand(batch_size, d, d_model_per_coord + (extra_dims > 0)).reshape(batch_size, -1)[:, :self.d_output]
        
        # Compute the sine and cosine for each batch
        pe = torch.zeros((batch_size, self.d_output), device=coords.device)
        pe[:, 0::2] = torch.sin(coords_expanded[:, 0::2] * div_term[0::2])
        pe[:, 1::2] = torch.cos(coords_expanded[:, 1::2] * div_term[1::2])
        
        pe = rearrange(pe, '(b n) d -> b n d', b=b, n=n)
        return pe
    






class PosEncoder(nn.Module):
    def __init__(self, config=None, perceiver_config=None, vision_width=768, num_frames=16, num_heads=8, token_out_channel=64):
        super(PosEncoder, self).__init__()
        self.config = edict(config)
        self.perceiver_config = edict(perceiver_config)
        self.vision_width = vision_width
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.token_out_channel = token_out_channel
        self.build_model()
        
        
    def build_model(self):
        logger.info("building position encoder")
        if self.config.model_type == 'perceiver':
            input_dim = 4 if self.config.use_bounding_box else 2
            self.coord_encoder = SinuPosEncoding(d_coords=input_dim, d_output=self.token_out_channel)
            self.motion_token_mapper = PerceiverResampler(
                dim=self.token_out_channel, 
                depth=self.perceiver_config.depth, 
                dim_head=self.token_out_channel // self.num_heads,
                heads=self.num_heads,
                num_latents=self.perceiver_config.num_latent * self.vision_width // self.token_out_channel , 
                use_rotary=self.perceiver_config.use_rotary,
                use_latent_transformer=self.perceiver_config.use_latent_transformer
            )
    
        elif self.config.model_type == 'mlp':
            input_dim = 4*self.num_frames if self.config.use_bounding_box else 2*self.num_frames
            self.motion_token_mapper = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=self.vision_width//4),  
                nn.ReLU(),  
                nn.Linear(in_features=self.vision_width//4, out_features=self.vision_width),  
            )
        elif self.config.model_type == 'sincos':
            input_dim = 4*self.num_frames if self.config.use_bounding_box else 2*self.num_frames
            self.motion_token_mapper = SinuPosEncoding(d_coords=input_dim, d_output=self.vision_width)
        else:
            raise NotImplementedError

    def forward(self, video_graph, segmasks):
        if len(segmasks.shape)==4: segmasks = decompose_masks(segmasks)
        bs,T,N,w,h = segmasks.shape
        d = 4 if self.config.use_bounding_box else 2
        if self.config.use_bounding_box:
            coords = compute_bounding_boxes(segmasks.reshape(-1,w,h), normalize=True).reshape(bs,T,N,-1)  # (bs,T,N,4)
        else:
            coords = compute_centers(segmasks.reshape(-1,w,h), normalize=True).reshape(bs,T,N,-1)  # (bs,T,N,2)
        
        coords = repeat(coords, "b t n d -> b m t n d", m=video_graph.shape[1])
        video_graph_m = repeat(video_graph, 'b m t -> b m t 1 d', d=d)
        graph_coords = torch.gather(coords, -2, video_graph_m)[..., 0,:] # (b,M,T,d)
        graph_coords_flat = rearrange(graph_coords, 'b M t d -> (b M) t d')
            
        padding_mask = rearrange(video_graph, 'b M t -> (b M) t') == 0
        # put a dummy 0 value for sequence that have all "1" value
        # it shuold be fine because transformer's attention mask will handle it.
        all_ones_rows = torch.all(padding_mask == 1, dim=1)  # This gives a boolean mask of rows with all 1s
        padding_mask[all_ones_rows, :10] = 0
            
        if self.config.model_type == 'perceiver':
            
            coord_embeds = self.coord_encoder(graph_coords_flat)
            mapped_tokens = self.motion_token_mapper(coord_embeds, padding_mask.float())
            mapped_tokens = rearrange(mapped_tokens, '(b M) (n N) d -> b M n (N d)', b=bs, M=video_graph.shape[1], n=self.perceiver_config.num_latent, N=self.vision_width // self.token_out_channel)
            mapped_tokens = torch.sum(mapped_tokens, dim=2)
        else:
            graph_coords = rearrange(graph_coords, 'b M t d -> b M (t d)')
            mapped_tokens = self.motion_token_mapper(graph_coords) 
            
        return mapped_tokens
        
            
        
            
class SegPosEncoder(nn.Module):
    def __init__(self, config=None, perceiver_config=None, vision_width=768, num_frames=16):
        super(SegPosEncoder, self).__init__()
        self.config = edict(config)
        self.perceiver_config = edict(perceiver_config)
        self.vision_width = vision_width
        self.num_frames = num_frames
        self.build_model()
        
        
    def build_model(self):
        logger.info("building position encoder")
        input_dim = 5 if self.config.use_bounding_box else 3
        self.motion_token_mapper = SinuPosEncoding(d_coords=input_dim, d_output=input_dim * 60)
        self.embed_mapper = nn.Linear(input_dim * 60, self.vision_width)
        
    def xy_to_xyt(self, coordinates):
        b, T, N, _ = coordinates.shape
        # Create a time tensor that ranges from 0 to T-1 for each frame
        time_tensor = torch.arange(self.num_frames, device=coordinates.device).float()  # Shape (T,)
        
        if T > self.num_frames:
            if T % self.num_frames == 0:
                time_tensor = time_tensor[:,None].repeat(1, T//self.num_frames).reshape(-1)
            else:
                raise NotImplementedError
            
        time_tensor = time_tensor.view(1, T, 1, 1).expand(b, T, N, 1)  # Shape (b, M, T, 1)

        # Concatenate the x, y coordinates with the time tensor
        xy_time_coordinates = torch.cat([coordinates, time_tensor], dim=-1)  # Shape (b, M, T, 3)
        
        return xy_time_coordinates
    
    def forward(self, segmasks):
        bst,N,w,h = segmasks.shape
        T = self.num_frames
        bs = bst // T
        d = 4 if self.config.use_bounding_box else 2
        if self.config.use_bounding_box:
            coords = compute_bounding_boxes(segmasks.reshape(-1,w,h), normalize=True).reshape(bs,T,N,-1)  # (bs,t,N,4)
        else:
            coords = compute_centers(segmasks.reshape(-1,w,h), normalize=True).reshape(bs,T,N,-1)  # (bs,t,N,2)
        
        coords = self.xy_to_xyt(coords) # (bs,t,N,2)
        coords = rearrange(coords, 'b t N d -> (b t) N d')
        
        sinu_embed = self.motion_token_mapper(coords)
        mapped_tokens = self.embed_mapper(sinu_embed)
            
        return mapped_tokens
    
    
    
    
    
class PatchPosEncoder(nn.Module):
    def __init__(self, config=None, perceiver_config=None, vision_width=768, num_frames=16):
        super(PatchPosEncoder, self).__init__()
        self.config = edict(config)
        self.perceiver_config = edict(perceiver_config)
        self.vision_width = vision_width
        self.num_frames = num_frames
        self.build_model()
        
        
    def build_model(self):
        logger.info("building position encoder")
        if self.config.model_type == 'perceiver':
            input_dim = 1
            if self.perceiver_config.version != 'v0':
                self.coord_encoder = SinuPosEncoding(d_coords=input_dim, d_output=self.vision_width,)
                self.motion_token_mapper = PerceiverResampler(
                    dim=self.vision_width, 
                    depth=self.perceiver_config.depth, 
                    dim_head=self.vision_width // 8,
                    heads=8,
                    num_latents=self.perceiver_config.num_latent, 
                    use_rotary=self.perceiver_config.use_rotary
                )
            else:
                self.motion_token_mapper = PerceiverResamplerv0(latent_dim=self.vision_width, num_latents=1, input_dim=input_dim, max_seq_len=self.num_frames) 
    
        elif self.config.model_type == 'mlp':
            input_dim = 4*self.num_frames if self.config.use_bounding_box else 2*self.num_frames
            self.motion_token_mapper = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=self.vision_width//4),  
                nn.ReLU(),  
                nn.Linear(in_features=self.vision_width//4, out_features=self.vision_width),  
            )
        elif self.config.model_type == 'sincos':
            input_dim = 4*self.num_frames if self.config.use_bounding_box else 2*self.num_frames
            self.motion_token_mapper = SinuPosEncoding(d_coords=input_dim, d_output=self.vision_width)
        else:
            raise NotImplementedError

    def forward(self, video_graph, segmasks):
        bs = video_graph.shape[0]
        
        video_graph_m = repeat(video_graph, 'b m t -> b m t 1')
        
        graph_coords_flat = rearrange(video_graph_m, 'b M t d -> (b M) t d')
            
        padding_mask = rearrange(video_graph, 'b M t -> (b M) t') == 0
        # put a dummy 0 value for sequence that have all "1" value
        # it shuold be fine because transformer's attention mask will handle it.
        all_ones_rows = torch.all(padding_mask == 1, dim=1)  # This gives a boolean mask of rows with all 1s
        padding_mask[all_ones_rows, :10] = 0
            
        if self.config.model_type == 'perceiver':
            
            if self.perceiver_config.version != 'v0':
                coord_embeds = self.coord_encoder(graph_coords_flat)
                mapped_tokens = self.motion_token_mapper(coord_embeds, padding_mask.float())
                mapped_tokens = rearrange(mapped_tokens, '(b M) n d -> b (M n) d', b=bs, M=video_graph.shape[1], n=self.perceiver_config.num_latent)
            else:
                mapped_tokens = self.motion_token_mapper(graph_coords_flat, padding_mask.float())
                mapped_tokens = rearrange(mapped_tokens[:, 0, :], '(b M) d -> b M d', b=bs, M=video_graph.shape[1])
        else:
            graph_coords = rearrange(graph_coords, 'b M t d -> b M (t d)')
            mapped_tokens = self.motion_token_mapper(graph_coords) 
            
        return mapped_tokens
        
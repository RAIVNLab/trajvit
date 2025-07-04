import torch
from torch import nn
from einops import rearrange, repeat
from easydict import EasyDict as edict
from transformers import ViTConfig
import logging
import numpy as np
import torch.nn.functional as F


from share_models.utils.resnet import CustomResNet
from share_models.utils.perceiver_resampler import PerceiverResampler
from share_models.traj_pos_embed import PosEncoder
from share_models.utils.tensors import decompose_masks, print_gpu_memory, apply_masks
from share_models.utils.vit_attn import ViTModel


logger = logging.getLogger(__name__)
# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class CustomTransformer(nn.Module):
    def __init__(
        self, 
        model_name="vit-base",  
        emb_dropout=0., 
        pool=None, 
        pretrained=True,
        new_depth=None,
        embed_dim=768,
    ):
        super().__init__()
        patch = 16 if model_name!='vit-huge' else 14
        model_name = f"google/{model_name}-patch{patch}-224-in21k"
        
        # Load a pretrained ViT model and remove the patch embedding layers
        config = ViTConfig.from_pretrained(model_name)
        self.embed_dim = config.hidden_size = embed_dim
        config.intermediate_size = embed_dim * 4
        self.num_heads = config.num_attention_heads
        logger.info(f"embedding size {self.embed_dim}")
        
        config.use_x_former = False
        if new_depth is not None: config.num_hidden_layers = new_depth  
        if not pretrained: 
            self.vit = ViTModel(config) 
            logger.info(f"loading vit {model_name} from scratch")
        else:
            self.vit = ViTModel.from_pretrained(model_name)
            logger.info(f"loading pretrained vit: {model_name}")
        
        # Adapt positional embeddings and CLS token to sequence length
        if pool is not None: self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.pool = pool


    def forward(self, x, attention_mask=None, output_attention=False):
        b, n, _ = x.shape

        # Append CLS token and positional embedding
        if self.pool is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Modify the attention mask to account for CLS token
        if attention_mask is not None:
            cls_attention_mask = torch.ones(b, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((cls_attention_mask, attention_mask), dim=1)

        # Forward pass through the transformer
        x = self.vit.encoder(x, attention_mask=attention_mask)[0]  # Get hidden states

        # Pooling based on specified method
        pooled_x = x.mean(dim=1, keepdim=True) if self.pool == 'mean' else x[:, :1]
        
            
        # Forward pass through the transformer with attentions
        outputs = self.vit.encoder(
            x, 
            attention_mask=attention_mask, 
            output_attentions=output_attention
        )
        
        hidden_states = outputs[0]  # Hidden states
        attention_scores = outputs[1] if output_attention else None

        # Pooling based on specified method
        pooled_x = hidden_states.mean(dim=1, keepdim=True) if self.pool == 'mean' else hidden_states[:, :1]
        
        if output_attention: return hidden_states, pooled_x, attention_scores
        else: return hidden_states, pooled_x
        
    
 

class VideoTokenViT(nn.Module):
    def __init__(self, config=None, pos_config=None, perceiver_config=None,  num_frames=16, norm_layer=nn.LayerNorm, use_fast_masking=False):
        super(VideoTokenViT, self).__init__()
        
        self.config = edict(config)
        self.perceiver_config=edict(perceiver_config)

        if self.config.model_name == 'vit-large':
            self.perceiver_config.depth = 4
            # self.config.tokenizer_type = 'resnet_50'
        
        self.token_out_channel = self.embed_dim = self.config.embed_dim # TODO
        self.num_frames = num_frames
        self.resnet_pool = self.config.get('resnet_pool', 'sum')
        assert self.num_frames == 16
        
        
        
        self.vision_encoder, self.vision_layernorm = self.build_vision_encoder()   
        self.num_heads = self.vision_encoder.num_heads
        
        self.build_vision_tokenizer()
        self.pos_encoder = PosEncoder(
            pos_config,
            perceiver_config,
            vision_width=self.embed_dim,
            num_frames=num_frames,
            num_heads=self.num_heads,
            token_out_channel=self.token_out_channel
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = nn.Identity()
        
    def build_vision_tokenizer(self):
        tokenizer_type = self.config.get('tokenizer_type', 'resnet')
        if tokenizer_type == 'resnet':
            logger.info("initializing ResNet tokenizer")
            self.appearance_tokenizer = CustomResNet(
                model_name="resnet-18",
                out_channel=self.token_out_channel,
                upsample_then_downsample=True,
                pretrained=self.config.pretrained,
            )
        elif tokenizer_type == 'resnet-50' or 'resnet_50' or 'resnet50':
            logger.info("initializing ResNet50 tokenizer")
            self.appearance_tokenizer = CustomResNet(
                model_name="resnet-50",
                out_channel=self.token_out_channel,
                upsample_then_downsample=True,
                pretrained=self.config.pretrained,
            )
        # elif tokenizer_type == 'hiera':
        #     logger.info("initializing Hiera tokenizer")
        #     self.appearance_tokenizer = CustomHiera(
        #         model_name="hiera_tiny",
        #         out_channel=self.token_out_channel,
        #         upsample_then_downsample=True,
        #         pretrained=self.config.pretrained
        #     )
        else:
            raise NotImplementedError
        if self.config.app_perceiver: 
            logger.info(f"perceiver uses latent transformer: {self.perceiver_config.use_latent_transformer}")
            logger.info("perceiver concat latent")
            self.appearance_token_mapper = PerceiverResampler(
                dim=self.token_out_channel, 
                depth=self.perceiver_config.depth, 
                dim_head=self.token_out_channel // self.num_heads,
                heads=self.num_heads,
                num_latents=self.perceiver_config.num_latent * self.embed_dim // self.token_out_channel, 
                use_rotary=self.perceiver_config.use_rotary,
                use_latent_transformer=self.perceiver_config.use_latent_transformer
            )
            
                
        else: self.appearance_token_mapper = torch.nn.Linear(self.token_out_channel, self.embed_dim)
        
        
    def freeze_vision_encoder(self):
        logger.info("calling freeze vision encoder")
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_vision_encoder(self):
        logger.info("calling UNfreeze vision encoder")
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
            
    def build_vision_encoder(self):
        logger.info("building ViTToken model")
        vision_encoder = CustomTransformer(
            model_name=self.config.model_name,
            pretrained=self.config.pretrained,
            pool=self.config.pool,
            embed_dim=self.config.embed_dim
        )
        return vision_encoder, None


    def masked_pooling(self, features: torch.Tensor,
                            masks:    torch.Tensor,
                        ) -> torch.Tensor:
        """
        features : (B,T,1,H,W,D)   fp16/fp32
        masks    : (B,T,N,H,W)     bool / int
        returns  : (B,T,N,D)
        """
        assert features.shape[2] == 1, "expected singleton channel dim"

        b, t, _, h, w, d = features.shape
        BT, L, N = b*t, h*w, masks.size(2)

        # ── flatten spatial dims ───────────────────────────────
        feats2d = features.view(BT, L, d)                 # (BT, L, D)
        masks2d = masks.view(BT, N, L).permute(0, 2, 1).float()   # (BT, L, N)

        cnt = masks2d.sum(1).clamp(min=1)       # (BT, N)  **float**
        masks2d = masks2d / cnt.unsqueeze(1)               # normalise
        
        
        # ── accumulate ────────────────────────────────────────
        pooled = torch.matmul(feats2d.transpose(1, 2),        # (BT, D, L)
                        masks2d)                # (BT, L, N)
        pooled = pooled.permute(0, 2, 1)                  # (BT, N, D)

        # keep output in fp32 unless you *really* need fp16
        return pooled.view(b, t, N, d)



    def _masked_roi_align(self, features, masks, out_size):
        B, T, _, H, W, C = features.shape
        N = masks.shape[2]
        
        masks = masks.unsqueeze(-1)
        masked = features * masks.float()
        
        masked = masked.view(B * T * N, H, W, C).permute(0, 3, 1, 2)  # (B*T*N, C, H, W)

        mask2d = masks[..., 0]   # (B, T, N, H, W)
        mask2d = mask2d.view(B * T * N, H, W).bool()  # (B*T*N, H, W)

        yy = torch.arange(H, device=features.device).view(1, H, 1).expand(B * T * N, H, W)
        xx = torch.arange(W, device=features.device).view(1, 1, W).expand(B * T * N, H, W)
        
        ymin = torch.where(mask2d, yy, torch.full_like(yy, H)).view(B * T * N, -1).min(dim=1)[0]  # (B*T*N,)
        ymax = torch.where(mask2d, yy, torch.zeros_like(yy)).view(B * T * N, -1).max(dim=1)[0]
        xmin = torch.where(mask2d, xx, torch.full_like(xx, W)).view(B * T * N, -1).min(dim=1)[0]
        xmax = torch.where(mask2d, xx, torch.zeros_like(xx)).view(B * T * N, -1).max(dim=1)[0]
        
        out_h, out_w = out_size, out_size
        N_rois = B * T * N
        device = features.device

        # Create a base grid in normalized coordinates in [0,1]
        grid_y_lin = torch.linspace(0, 1, out_h, device=device)  # (2,)
        grid_x_lin = torch.linspace(0, 1, out_w, device=device)   # (2,)
        grid_y, grid_x = torch.meshgrid(grid_y_lin, grid_x_lin, indexing='ij')  # each (2,2)
        # Expand to (N_rois, 2,2)
        grid_y = grid_y.unsqueeze(0).expand(N_rois, -1, -1)
        grid_x = grid_x.unsqueeze(0).expand(N_rois, -1, -1)

        # For each ROI, map the base grid to the pixel coordinates of the bounding box.
        # The pixel coordinate of a grid point is: grid*(max - min) + min.
        roi_y = grid_y * (ymax - ymin).view(-1, 1, 1) + ymin.view(-1, 1, 1)  # (N_rois, 2,2)
        roi_x = grid_x * (xmax - xmin).view(-1, 1, 1) + xmin.view(-1, 1, 1)  # (N_rois, 2,2)

        # grid_sample expects normalized coordinates in the range [-1, 1]. Normalize accordingly:
        roi_y_norm = 2 * roi_y / (H - 1) - 1
        roi_x_norm = 2 * roi_x / (W - 1) - 1

        roi_grid = torch.stack((roi_x_norm, roi_y_norm), dim=-1)  # (N_rois, 2,2,2)

        pooled = F.grid_sample(masked, roi_grid, align_corners=True)  # (B*T*N, C, 2, 2)
        pooled = pooled.view(B, T, N, C, out_h*out_w)
        concat_pooled = rearrange(pooled, 'b t n c hw -> b t n (hw c)')
        return concat_pooled
        

    def encode_appearance_token(self, video, masks, video_graph, get_intermediate_feature=False):
        if len(masks.shape)==4: masks = decompose_masks(masks)
        # masks = masks.unsqueeze(-1) # (b, t, N, w, h, d)
        bs, T = video.shape[0], video.shape[1]
        mask_W, mask_H = masks.shape[3], masks.shape[4]
        video_frames = rearrange(video, 'b t d w h -> (b t) d w h')
        
        out_features = self.appearance_tokenizer(video_frames, pool=self.resnet_pool, output_size=(mask_W, mask_H))
        out_features = rearrange(out_features, '(b t) d w h -> b t 1 w h d', b=bs, t=T)
        

        # avoid temporaily large memory reserve (not consumed) in this step 
        # seems to be a torch library issue -- call export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" helps       
        step = 1024 * 16 * 64  // (T * bs)
        masked_features = []
        for d in range(0, out_features.shape[-1], step):
            out_features_c = out_features[..., d:d+step]
            if self.config.get('roi_size', 0) > 0:
                masked_features.append(self._masked_roi_align(out_features_c, masks, out_size=self.config.roi_size))  # (b t N d) 
            else:
                masked_features.append(self.masked_pooling(out_features_c, masks))  # (b t N d) 
        masked_features = torch.cat(masked_features, dim=-1)
        masked_features_c = repeat(masked_features, 'b t N d -> b M t N d', M=video_graph.shape[1])
        video_graph_c = repeat(video_graph, 'b M t -> b M t 1 d', d=masked_features.shape[-1])
        per_frame_graph_feature = torch.gather(masked_features_c, -2, video_graph_c)[..., 0, :]  # (b,M,t,d) # avoid using squeeze
        if get_intermediate_feature: 
            return out_features[:,:,0], per_frame_graph_feature   # (b t w h d)   (b M t d)
        
        if self.config.app_perceiver:
            graph_feature_flat = rearrange(per_frame_graph_feature, 'b M t d -> (b M) t d')
            
            padding_mask = rearrange(video_graph, 'b M t -> (b M) t') == 0
            # put a dummy 0 value for sequence that have all "1" value
            # it shuold be fine because transformer's attention mask will handle it.
            all_ones_mask = torch.all(padding_mask == 1, dim=1)  # This gives a boolean mask of rows with all 1s
            padding_mask[all_ones_mask, :10] = 0
            
            if self.config.get('roi_size', 0) > 0:
                graph_feature_flat = rearrange(graph_feature_flat, 'b n (four d) -> b (four n) d', four=self.config.roi_size**2)
                padding_mask = repeat(padding_mask, 'b n -> b (four n)', four=self.config.roi_size**2)
            
            mapped_tokens = self.appearance_token_mapper(graph_feature_flat, padding_mask.float())
            mapped_tokens = rearrange(mapped_tokens, '(b M) (n N) d -> b M n (N d)', b=bs, M=video_graph.shape[1], n=self.perceiver_config.num_latent, N=self.embed_dim // self.token_out_channel)
            mapped_tokens = torch.sum(mapped_tokens, dim=2)
        else:
            app_tok_features = torch.sum(per_frame_graph_feature, dim=2)  # (b,M,d)
            num_frames_per_token = torch.sum(video_graph>0, dim=-1, keepdim=True).clamp(min=1)
            app_tok_features = app_tok_features / num_frames_per_token
            mapped_tokens = self.appearance_token_mapper(app_tok_features)
        
        return mapped_tokens
            
    
    
    def forward(self, video, masks=None, segmask=None, video_graph=None, output_attention=False):
        """main model forward

        Args:
            video (tensor[b,c,t,w,h])
            maemask (_type_): a list of valid index

        Returns:
            tensor[(bs, L, d)]: output traj embedding
        """
        if video.shape[1] == 3 : # default t channel in dim 1
            video = rearrange(video, "b c t w h -> b t c w h")
        else:
            assert video.shape[2] == 3
    
        if masks is not None:
            video_graph = apply_masks(video_graph, masks)
        
        app_tokens = self.encode_appearance_token(video, segmask, video_graph)
        pos_tokens = self.pos_encoder(video_graph, segmask)
        x = app_tokens + pos_tokens
        
        traj_graph_sum = torch.sum(video_graph, dim=-1)
        attn_mask = traj_graph_sum != 0
        attn_mask = repeat(attn_mask, 'b n -> b (n m)', m=self.perceiver_config.num_latent)
        
        if not output_attention:
            x, _ = self.vision_encoder(x, attn_mask.float(), output_attention=output_attention)
        else:
            x, _, attn_score = self.vision_encoder(x, attn_mask.float(), output_attention=output_attention)
            
        x = self.norm(x)

        if output_attention: return x, attn_score
        else: return x  # (bs, L, d)    


def count_parameters(model, trainable_only: bool = True):
    """
    Args
    ----
    model : torch.nn.Module
    trainable_only : bool
        • True  → count only parameters with requires_grad = True  
        • False → count every parameter in the model

    Returns
    -------
    total_params : int
        Number of scalar parameters.
    """
    param_iter = (p for p in model.parameters()
                  if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in param_iter)





if __name__ == "__main__":
    config = {
        "model_name": "vit-large",
        "out_channel": 64,
        "app_perceiver": True,
        "pretrained": False,
        "pool": "cls",
        "hiera_feat_pool": "sum",
        "roi_size": 0,
        "resnet_pool": "sum",
        "embed_dim": 1024,
        "tokenizer_type": "resnet"
    }
    pos_config = {
        "model_type": "perceiver",  #  mlp, sincos
        "use_bounding_box": True,
    }
    perceiver_config = {
        "num_latent": 1,
        "depth": 2,
        "use_rotary": False,
        "use_latent_transformer": False,
        "concat_latent": False,
    }
    model = VideoTokenViT(config, pos_config, perceiver_config)
    
    trainable = count_parameters(model, trainable_only=False)
    print(trainable)
    from calflops import calculate_flops

    # # Define a wrapper function to handle additional inputs
    # class WrappedModel(torch.nn.Module):
    #     def __init__(self, model, segmask, video_graph):
    #         super().__init__()
    #         self.model = model
    #         self.segmask = segmask
    #         self.video_graph = video_graph

    #     def forward(self, input_video):
    #         return self.model(input_video, masks=None, segmask=self.segmask, video_graph=self.video_graph)
    
    # import json
    # # with open('/weka/chenhaoz/home/videotok/parse_flops/token_number.json') as f:
    # #     token_nums = json.load(f)['ours_k700']
        
    # token_nums = [100,150,200]
    # data = {}
    # flops = 0
    # for model_name in [ 'vit-base', 'vit-large',  'vit-huge',]:
    #     flop_list = []
    #     for tok_idx, frame_number in enumerate([16, 32, 64]):
    #         # Initialize your model and inputs
    #         B, T, H, W = 1, frame_number, 224, 224  # Batch size, frames, height, width
    #         N_traj, N_max_mask = int(token_nums[tok_idx]), 64
            
    #         config['model_name'] = model_name

    #         model = VideoTokenViT(config=config, pos_config=pos_config, num_frames=T).cuda()

    #         input_video = torch.randn(B, 3, T, H, W).cuda()  # Input video tensor
    #         segmask = torch.zeros(B, T, N_max_mask, H // 4, W // 4, dtype=torch.bool).cuda()  # Segmentation mask
    #         video_graph = torch.zeros(B, N_traj, T, dtype=torch.int64).cuda()  # Video graph

    #         # Wrap the model to include additional inputs
    #         wrapped_model = WrappedModel(model, segmask, video_graph)

    #         input_shape = (B, 3, T, H, W)
    #         flops, macs, params = calculate_flops(
    #             model= wrapped_model,
    #             input_shape=input_shape,
    #             output_as_string=False,
    #             output_precision=4,
    #             print_results=False,
    #             print_detailed=False
    #         )
            
    #         flop_list.append(flops / 1e9)
    #     data[model_name] = flop_list
    
    # with open('/weka/chenhaoz/home/videotok/parse_flops/model_stats_vittok.json', 'w') as f:
    #     json.dump(data, f, indent=4)
        
        


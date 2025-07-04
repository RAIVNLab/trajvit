import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import hiera

class CustomHiera(nn.Module):
    def __init__(self, model_name="hiera_tiny", upsample_then_downsample=True, 
                 out_channel=512, pretrained=False):
        super(CustomHiera, self).__init__()
        

        self.hiera = hiera.hiera_tiny_224(pretrained=pretrained, checkpoint="mae_in1k_ft_in1k")
        assert  model_name == 'hiera_tiny'
        
        self.upsample_then_downsample = upsample_then_downsample

        # Retrieve layer output sizes from the loaded config
        hidden_sizes = [96, 192, 384, 768]
        
        # Define linear layers for each stage's output, if upsampling then downsampling
        if self.upsample_then_downsample:
            self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, out_channel) for hidden_size in hidden_sizes
            ])

    def forward(self, x, pool='sum', output_size=(224,224)):
        # Get intermediate layer outputs
        _, hidden_states = self.hiera(x, return_intermediates=True)

        # Apply linear layers if upsample_then_downsample is True
        if self.upsample_then_downsample:
            for i, feature in enumerate(hidden_states):
                feature = self.linear_layers[i](feature)
                hidden_states[i] = rearrange(feature, 'b w h d -> b d w h')

        # Resize all features to the specified output size
        resized_features = [F.interpolate(feature, size=output_size, mode='bilinear', align_corners=False)
                            for feature in hidden_states]

        # Pooling method
        if pool == 'sum':
            out_features = sum(resized_features)
        elif pool == 'concat':
            out_features = torch.cat(resized_features, dim=1)
        else: raise NotImplementedError
        
        return out_features
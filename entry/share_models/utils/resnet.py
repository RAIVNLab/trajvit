import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import ResNetModel, ResNetConfig

class CustomResNet(nn.Module):
    def __init__(self, model_name="resnet18", upsample_then_downsample=True, 
                 out_channel=512, pretrained=False):
        super(CustomResNet, self).__init__()
        
        # Load the ResNet model with specified configuration
        model_path = f"microsoft/{model_name}"
        config = ResNetConfig.from_pretrained(model_path)
        
        # Load model with specified configuration
        self.resnet = ResNetModel(config) if not pretrained else ResNetModel.from_pretrained(model_path, config=config)
        
        self.upsample_then_downsample = upsample_then_downsample

        # Retrieve layer output sizes from the loaded config
        hidden_sizes = self.resnet.config.hidden_sizes
        self.out_channel = out_channel
        
        # Define linear layers for each stage's output, if upsampling then downsampling
        if self.upsample_then_downsample:
            self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, out_channel) for hidden_size in hidden_sizes
            ])
        
        self.layer_norm = nn.LayerNorm((out_channel))

    def forward(self, x, pool='sum', output_size=(56,56)):
        # Get intermediate layer outputs
        outputs = self.resnet(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of tensors from each stage

        # Extract the desired stages
        stage_outputs = [hidden_states[i] for i in [1, 2, 3, 4]]  # Typically corresponds to stages 1-4

        # Apply linear layers if upsample_then_downsample is True
        if self.upsample_then_downsample:
            for i, feature in enumerate(stage_outputs):
                feature = rearrange(feature, 'b d w h -> b w h d')
                feature = self.linear_layers[i](feature)
                feature = self.layer_norm(feature)
                stage_outputs[i] = rearrange(feature, 'b w h d -> b d w h')

        # Resize all features to the specified output size
        resized_features = []
        for feature in stage_outputs:
            rf = F.interpolate(feature, size=output_size, mode='bilinear', align_corners=False)
            resized_features.append(rf)
        # resized_features = [F.layer_norm(feat, feat.size()[1:]) for feat in resized_features]
        # Pooling method
        if pool == 'sum':
            out_features = sum(resized_features)
        elif pool == 'concat':
            out_features = torch.cat(resized_features, dim=1)
        elif pool == 'lastlayer':
            out_features = resized_features[-1]
        else: raise NotImplementedError
        
        return out_features
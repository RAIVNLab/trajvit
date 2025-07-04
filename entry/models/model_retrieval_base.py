from models.utils import (
    interpolate_pos_embed,
    interpolate_pos_relative_bias_beit,
    _init_transformer_weights,
    interpolate_temporal_pos_embed
)
from transformers import BeitModel, BeitConfig, ViTModel, ViTConfig
from models.xbert import BertConfig, BertModel, BertForMaskedLM, BertEncoder
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import logging
import torch.distributed as dist
import torch.distributed.nn
from share_models.traj_transformer import CustomTransformer

logger = logging.getLogger(__name__)




class SingularityRetrievalBase(nn.Module):
    """Common utils shared by pretraining and downstream retrieval"""
    def __init__(self, config=None, tokenizer=None, pretrain=True, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_dim = config.embed_dim
        self.pretrain = pretrain
        self.text_width = 768 # TODO: have to be flexible

        self.vision_encoder, self.vision_layernorm = self.build_vision_encoder()
        
        self.text_encoder = self.build_text_encoder()
        

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.control_num = 0

        self.temp = nn.Parameter(torch.ones([]) * config.temp)
        self.itm_head = nn.Linear(self.text_width, 2)
    

    def forward(self, vision_input, text, idx):
        
        # ================= Dual Encoder ITC loss ================ #
        self.clip_contrastive_temperature()
        
        # print("input image shape:", image.shape) # (B, T, 3, 224, 224)
        image_embeds, pooled_image_embeds = self.encode_image(vision_input)
        
        text_embeds, pooled_text_embeds = self.encode_text(text)

        loss_ita, sim_i2t, sim_t2i, accuracy_ita = self.get_contrastive_loss(
            pooled_image_embeds, pooled_text_embeds, idx)
            
        return_dict = dict(
            loss_ita=loss_ita * self.config.loss_weight.itc,
            accuracy_ita = accuracy_ita
        )
        return return_dict
    

    def build_text_encoder(self):
        logger.info(f"Build text_encoder {self.config.text_encoder}")
        bert_config = BertConfig.from_json_file(self.config.bert_config)
        if self.pretrain:
            text_encoder, loading_info = BertForMaskedLM.from_pretrained(
                self.config.text_encoder, config=bert_config,
                output_loading_info=True
            )
        else:
            text_encoder, loading_info = BertModel.from_pretrained(
                self.config.text_encoder, config=bert_config,
                add_pooling_layer=False, output_loading_info=True
            )

        # if self.config.debug:
        #     for k, v in loading_info.items():
        #         logger.debug(f"loading_info[{k}]: {v}")
        logger.info(f"Build text_encoder {self.config.text_encoder}, done!")
        return text_encoder

    def build_vision_encoder(self):
        
        if self.config.vit_type in ["beit", "deit", "vit", 'vitmatch', 'vitlearner']:
            vision_encoder = self.build_huggingface_vit_with_image_size(
                self.config.vit_name_or_pretrained_path, self.config.image_res)
        else:
            raise ValueError(f"Unknown vit type {self.config.vit_type}")

        # add layernorm for normalizing BEiT outputs hidden states
        vision_layernorm = None
        if self.config.vit_type == "beit":
            vision_layernorm = nn.LayerNorm(self.vision_width, eps=1e-12)
        return vision_encoder, vision_layernorm



    def build_huggingface_vit_with_image_size(self, model_card: str, image_size: int, ):
        """Build a vit model from huggingface hub, also interpolate pos_embed when needed.

        Args:
            model_card: name in huggingface hub, e.g., `facebook/deit-base-patch16-224`
            image_size: new image size, may be different from pre-training image_size of `model_card`

        ref: https://github.com/huggingface/transformers/issues/12167#issuecomment-861356232
        """
        is_beit = "beit" in model_card
        if "beit" in model_card:
            model_cls, config_cls = BeitModel, BeitConfig
        elif "deit" in model_card or "vit" in model_card:
            # the deit model we use is loaded in vit arch,
            # see https://huggingface.co/facebook/deit-base-patch16-224#how-to-use
            model_cls, config_cls = ViTModel, ViTConfig
        else:
            raise ValueError(f"Unexpected model_card: {model_card}")

        logger.info(f"Init new model with new image size {image_size}, and load weights.")
        model_config = config_cls.from_pretrained(model_card, image_size=image_size)
        if hasattr(self, 'vision_width'): model_config.hidden_size = self.vision_width
        else: self.vision_width = model_config.hidden_size
        model = model_cls(config=model_config, add_pooling_layer=is_beit)
        # TODO: if pretrained
        return model

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def encode_image(self, vision_input, training_mask=None):  
        if type(vision_input) == tuple: image = vision_input[0]
        else: image = vision_input
        
        bsz, num_frms, c, h, w = image.shape  # `num_frms` could be changing for image (=1) or video (e.g., =4)

        image_embeds, pooled_image_embeds = self._encode_image(image)
        

        image_embeds = image_embeds.view(bsz, -1, self.vision_width)  # (bsz, num_frms*L, d)
        return image_embeds, pooled_image_embeds  # (bsz, #frm*L, d), (bsz, #frm, d)
    

    def _encode_image(self, vision_input):
        if type(vision_input) == tuple: image = vision_input[0]
        else: image = vision_input
        
        bsz, num_frms, c, h, w = image.shape  # `num_frms` could be changing for image (=1) or video (e.g., =4)
        image = image.view(bsz*num_frms, c, h, w)
        image_embeds = self.vision_encoder(image)
        if self.vision_layernorm is not None:  # only for BEiT mean-pooling
            image_embeds.last_hidden_state = self.vision_layernorm(image_embeds.last_hidden_state)

        if self.config.vit_type == "beit":
            pooled_image_embeds = image_embeds.pooler_output  # (bsz*num_frms, d)
            image_embeds = image_embeds.last_hidden_state  # (bsz*num_frms, L, d)
        else:
            image_embeds = image_embeds.last_hidden_state
            pooled_image_embeds = image_embeds[:, 0]

        image_embeds = image_embeds.view(bsz, num_frms, -1, self.vision_width)  # (bsz, num_frms, L, d)
        pooled_image_embeds = pooled_image_embeds.view(bsz, num_frms, self.vision_width) \
            if pooled_image_embeds is not None else None  # (bsz, num_frms, d)
        return image_embeds, pooled_image_embeds

    def encode_text(self, text):
        # text
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text'
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    @torch.no_grad()
    def get_mask(self, sim, idx=None, normalize=False):
        """
        sim: (N, N)
        idx: (N, )
        normalize: bool, make row sum equal to 1
        """
        if idx is not None:
            idx = idx.view(-1, 1)
            mask = torch.eq(idx, idx.T).to(sim.dtype)
            if normalize:  
                mask = mask / mask.sum(1, keepdim=True)
        else:
            mask = torch.zeros_like(sim)
            mask.fill_diagonal_(1)
        return mask  # `1` mark valid/matched location

    def get_contrastive_loss(self, pooled_image_embeds, pooled_text_embeds, idx=None):
        # Gather features from all GPUs
        
        sim_i2t, sim_t2i = self.get_sim(
            pooled_image_embeds, pooled_text_embeds, t=self.temp)
            
        with torch.no_grad():
            sim_i2t_targets = self.get_mask(sim_i2t, idx=idx, normalize=True)
            sim_t2i_targets = sim_i2t_targets

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        
        # Image-to-text accuracy
        i2t_preds = torch.argmax(sim_i2t, dim=1)  # Get the most similar text for each image
        i2t_correct = (i2t_preds == torch.arange(sim_i2t.size(0), device=sim_i2t.device)).sum().item()
        accuracy_i2t = i2t_correct / sim_i2t.size(0)
        
        # Text-to-image accuracy
        t2i_preds = torch.argmax(sim_t2i, dim=1)  # Get the most similar image for each text
        t2i_correct = (t2i_preds == torch.arange(sim_t2i.size(0), device=sim_t2i.device)).sum().item()
        accuracy_t2i = t2i_correct / sim_t2i.size(0)
        
        accuracy_ita = (accuracy_i2t + accuracy_t2i) / 2
        
        return loss_ita, sim_i2t, sim_t2i, accuracy_ita

    def get_sim(self, pooled_image_embeds, pooled_text_embeds, t=1, multi_gpu=True):
        """
        Args:
            pooled_image_embeds: (bsz, num_frms, d)
            pooled_text_embeds: (bsz, d) or (bsz, n, d) where n is template number
            t: temperature
        """
        image_proj = self.vision_proj
        text_proj = self.text_proj

        istemplate = False
        if len(pooled_text_embeds.shape) == 3:  # template that needs to be averaged
            istemplate = True
            n = pooled_text_embeds.shape[1]
            pooled_text_embeds = pooled_text_embeds.reshape(-1, pooled_text_embeds.shape[-1])

        image_feat = F.normalize(image_proj(pooled_image_embeds), dim=-1).contiguous()
        text_feat = F.normalize(text_proj(pooled_text_embeds), dim=-1).contiguous()
        
        if istemplate:
            text_feat = text_feat.reshape(-1,n,text_feat.shape[-1])
            text_feat = torch.mean(text_feat, dim=1)
        
        
        if multi_gpu:
            gathered_image_feat_tuple = torch.distributed.nn.functional.all_gather(image_feat)
            gathered_image_feat = torch.cat(gathered_image_feat_tuple, dim=0)
            gathered_text_feat_tuple = torch.distributed.nn.functional.all_gather(text_feat)
            gathered_text_feat = torch.cat(gathered_text_feat_tuple, dim=0)
            sim_i2t = torch.einsum("mld,nd->mln", gathered_image_feat, gathered_text_feat).mean(1) / t  # (N, N)
        
        else: sim_i2t = torch.einsum("mld,nd->mln", image_feat, text_feat).mean(1) / t  # (N, N)
        sim_t2i = sim_i2t.T
        return sim_i2t, sim_t2i

    def get_itm_loss(
            self, sim_i2t, sim_t2i, text_embeds, text_atts, image_embeds, image_atts, idx=None):
        """
        sim_i2t, sim_t2i: (N, N)
        text_embeds, text_atts, image_embeds, image_atts: (N, *)
        idx: (N, )
        """
        bsz = len(sim_i2t)
        text_encoder = self.get_text_encoder()

        with torch.no_grad():
            weights_i2t = F.softmax(sim_i2t+1e-4, dim=1)  # (N, N)
            weights_t2i = F.softmax(sim_t2i+1e-4, dim=1)

            mask = self.get_mask(sim_i2t, idx=idx).bool()
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        # select a negative image for each text
        if self.config.itm_hard_neg:
            img_neg_indices = torch.multinomial(weights_t2i, 1).squeeze()
        else:
            img_neg_indices = self.get_rand_indices(mask, 1).squeeze()            

        image_embeds_neg = image_embeds[img_neg_indices]

        # select a negative text for each image
        if self.config.itm_hard_neg:
            txt_neg_indices = torch.multinomial(weights_i2t, 1).squeeze()
        else:
            txt_neg_indices = self.get_rand_indices(mask, 1).squeeze()

        text_embeds_neg = text_embeds[txt_neg_indices]
        text_atts_neg = text_atts[txt_neg_indices]  # (N, L, d)

        # embedding on local gpu
        _text_embeds = text_embeds  
        _text_atts = text_atts
        _image_embeds = image_embeds
        _image_atts = image_atts
        # concat embeddings
        text_embeds_all = torch.cat([_text_embeds, _text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([_text_atts, _text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([_image_embeds, image_embeds_neg, _image_embeds], dim=0)
        image_atts_all = torch.cat([_image_atts, _image_atts, _image_atts], dim=0)
        output = text_encoder(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode='fusion',
        )
        
        itm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

        loss_itm = self._get_itm_loss(itm_embeds, enc=self.itm_head)
        itm_embeds_pos = itm_embeds[:bsz]  # (N, d)

        return loss_itm, itm_embeds_pos

    def _get_itm_loss(self, itm_embeds, enc):
        """
        itm_embeds: (3*N, D)
        enc: nn.Module that projects cls_embeds
        """
        itm_scores = enc(itm_embeds)  # (3*N, 2)
        bs = itm_scores.size(0) // 3
        itm_labels = itm_scores.new_ones(3*bs, dtype=torch.long)
        itm_labels[bs:] = 0
        loss_itm = F.cross_entropy(itm_scores, itm_labels)
        return loss_itm

    def get_rand_indices(self, mask, k):
        """
        Args:
            mask: (N, L) 0 indicates the positions that we can sample, 1 otherwise
            k: #indices to sample at each row
        Returns:
            (N, k) indices
        """
        mask = mask.float()
        mask = mask - 10000 * mask
        mask += torch.randn_like(mask)
        _, indices = torch.sort(mask, dim=1, descending=True)
        indices = indices[:, :k].contiguous()  
        return indices
    


import time
import datetime
import logging
import numpy as np

import torch
import torch.distributed as dist

from utils.basic_utils import MetricLogger
from utils.distributed import get_rank, get_world_size


logger = logging.getLogger(__name__)


def extract_text_feats(texts, max_txt_l, tokenizer, model, device):
    num_text = len(texts)
    if type(texts[0])==list: text_bs = 32
    else: text_bs = 256
    
    text_feats = []

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        if type(texts[0])==list:
            n = len(text[0])
            text = [item for sublist in text for item in sublist]
        text_input = tokenizer(
            text, padding="max_length",
            truncation=True, max_length=max_txt_l,
            return_tensors="pt"
        ).to(device)
        text_feat = model.encode_text(text_input)[0][:, 0]
        if type(texts[0])==list:
            text_feat = torch.stack([text_feat[i:i + n] for i in range(0, len(text_feat), n)])
        
        text_feats.append(text_feat.cpu())
        

    text_feats = torch.cat(text_feats, dim=0)
    return text_feats


def extract_vision_feats(data_loader, model, device, config):
    pooled_image_feats_all = []
    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting image feats"
    iterator = metric_logger.log_every(data_loader, 100, header)
    for image, text, idx, segment, graph, num_tokens in iterator:
        # TODO: change input
        image = image.to(device, non_blocking=True)
        
        # if config.vit_type.startswith('vittok'):
        segment = segment.to(device, non_blocking=True)
        graph = graph.to(device, non_blocking=True)
        num_tokens = num_tokens.to(device, non_blocking=True)

        if config.eval_frame_ensemble == "concat":  # default
            _, pooled_image_feat = model.encode_image((image, segment, graph, num_tokens))   # (bsz, #frm*L, d), (bsz, #frm, d)
            # except Exception as e: 
            #     pooled_image_feat = pooled_image_feats_all[-1]
            #     print("\tException!", e)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.eval_frame_ensemble in ["mean", "max", "lse"]
            _, pooled_image_feat = model._encode_image((image, segment, graph, num_tokens))   # (bsz, #frm, L, d), (bsz, #frm, d)
            
        pooled_image_feats_all.append(pooled_image_feat.cpu())
            
    # image_feats_all = torch.cat(image_feats_all, dim=0)
    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)
    return pooled_image_feats_all



# @torch.no_grad()
# def evaluation_wrapper(model, data_loader, tokenizer, device, config, prefix=""):
#     with torch.cuda.amp.autocast(enabled=config.fp16):
#         eval_func = cross_encoder_evaluation if config.eval_x_only else evaluation
#         i2t_x, t2i_x, i2t_emb, t2i_emb = eval_func(model, data_loader, tokenizer, device, config)
#     score_pairs = [
#         (prefix + "/", i2t_x, t2i_x),
#         (prefix + "_emb/", i2t_emb, t2i_emb),
#     ]
#     res = dict()
#     for name, i2t, t2i in score_pairs:
#         if i2t is not None:
#             txt2img_ids = data_loader.dataset.txt2img
#             img2txt_ids = data_loader.dataset.img2txt
#             res[name] = itm_eval(i2t, t2i, txt2img_ids, img2txt_ids)
#     return res

@torch.no_grad()
def evaluation_wrapper(model, data_loader, tokenizer, device, config, prefix=""):
    with torch.cuda.amp.autocast(enabled=config.fp16):
        i2t_emb, t2i_emb = evaluation(model, data_loader, tokenizer, device, config)
    score_pairs = [
        (prefix + "_emb/", i2t_emb, t2i_emb),
    ]
    res = dict()
    for name, i2t, t2i in score_pairs:
        if i2t is not None:
            txt2img_ids = data_loader.dataset.txt2img
            img2txt_ids = data_loader.dataset.img2txt
            res[name] = itm_eval(i2t, t2i, txt2img_ids, img2txt_ids)
            
    return res


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.fp16 else torch.float
    media_type = data_loader.dataset.media_type
    logger.info(f"Start evaluation for media_type={media_type}")

    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    texts = data_loader.dataset.text
    max_txt_l = config.max_txt_l
    if not isinstance(max_txt_l, int):
        max_txt_l = max_txt_l[media_type]
    pooled_text_feats = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device)  # (bsz, Lt, d), (bsz, Lt)
    pooled_image_feats = extract_vision_feats(
        data_loader, model, device, config)  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    logger.info("Finished feature extraction")
    logger.info("Computing ITC scores [dot-product]")
    pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)
    pooled_text_feats =  pooled_text_feats.to(device, non_blocking=True)
    i2t_scores, t2i_scores = model.get_sim(pooled_image_feats, pooled_text_feats, multi_gpu=False)
    logger.info("Computing ITC scores [dot-product], done!")
    
    return i2t_scores.cpu().numpy(), t2i_scores.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        gt_txt_ids = img2txt[index]
        if isinstance(gt_txt_ids, int):
            ranks[index] = np.where(inds == gt_txt_ids)[0][0]
        else:
            rank = 1e20
            for i in gt_txt_ids:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        gt_img_ids = txt2img[index]
        if isinstance(gt_img_ids, int):
            ranks[index] = np.where(inds == gt_img_ids)[0][0]
        else:  # list, used in the case each caption has multiple GT images
            # Score
            rank = 1e20
            for i in gt_img_ids:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)


    eval_result = {"img2txt_r1": tr1,
                   "img2txt_r5": tr5,
                   "txt2img_r1": ir1,
                   "txt2img_r5": ir5,}
    eval_result = {k: round(v, 2) for k, v in eval_result.items()}
    return eval_result

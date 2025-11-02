import collections
import os
from prettytable import PrettyTable

import torch
import numpy as np
import random
import json

from datasets import build_dataloader
from tqdm import tqdm
from model import build_model
from utils.checkpoint import Checkpointer
from utils.options import get_args
from utils.comm import get_rank, synchronize
import torch.nn.functional as F

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity.data.cpu(), dim=1, descending=True)
        indices = indices.to(similarity.device)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, captions, img_paths = [], [], [], [], [], []
        # text
        for pid, caption_id, caption in self.txt_loader:
            caption_id = caption_id.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption_id)
            qids.append(pid.view(-1))  # flatten
            qfeats.append(text_feat.data.cpu())
            captions.extend(caption)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img, img_path in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat.data.cpu())
            img_paths.extend(img_path)
        gids = torch.cat(gids, 0)

        gfeats = torch.cat(gfeats, 0)

        return qfeats.cuda(), gfeats.cuda(), qids, gids, captions, img_paths

    def eval(self, model, save_path="retrieval_dataset.json", i2t_metric=False):

        qfeats, gfeats, qids, gids, captions, img_paths = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

        similarity = qfeats @ gfeats.t()
        results = []
        num_queries = len(captions)

        for i in tqdm(range(num_queries), desc="Building retrieval dataset"):
            qid = qids[i].item()
            sims = similarity[i]  # (num_gallery,)
            topk_indices = torch.topk(sims, k=5).indices.tolist()
            topk_paths = [img_paths[j] for j in topk_indices]
            topk_ids = [gids[j].item() for j in topk_indices]

            # Case 1: 前5中存在相同ID
            if qid in topk_ids:
                real_idx = topk_ids.index(qid)+1
            else:
                # Case 2: 没有相同ID，找全局相似度最高且ID相同的图像
                same_id_mask = (gids == qid)
                if same_id_mask.any():
                    same_id_sims = sims[same_id_mask]
                    global_best_idx = torch.argmax(same_id_sims).item()
                    global_indices = torch.where(same_id_mask)[0]
                    global_best_gallery_idx = global_indices[global_best_idx].item()
                    real_img_path = img_paths[global_best_gallery_idx]
                    # 随机替换前5中的一个
                    replace_idx = random.randint(0, 4)
                    topk_paths[replace_idx] = real_img_path
                    real_idx = replace_idx+1
                else:
                    real_idx = 0

            results.append({
                "caption": captions[i],
                "images": topk_paths,
                "real_image": real_idx  # 改成索引位置
            })

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Retrieval dataset saved to {save_path}")

        return results


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1 + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"

    args.training = False

    train_img_loader, train_txt_loader, test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load('./checkpoint/best0.pth')
    model.to(device)

    evaluator = Evaluator(train_img_loader, train_txt_loader)
    top1 = evaluator.eval(model.eval(),save_path='./top_data/train_top5_data.json')


    test_model = build_model(args, num_classes=68126)
    checkpointer = Checkpointer(test_model)
    checkpointer.load('./20251101_211148_finetune/best0.pth')
    test_model.to(device)

    evaluator1 = Evaluator(test_img_loader, test_txt_loader)
    top2 = evaluator1.eval(test_model.eval(),save_path='./top_data/test_top5_data.json')




import argparse
import os
import json
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, get_scheduler
import wandb
from peft import LoraConfig, get_peft_model
from data import DocVQADataset,DocVQADataset_test, PedestrianMatchingDataset, PedestrianMatchingDataset_Test

from multilabel_metrics import AveragePrecisionMeter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sys, re

import re


def normalize_answer(answer: str) -> str:
    """
    标准化答案，提取核心关键词并统一格式
    """
    # 转小写并去除首尾空格
    answer = answer.strip().lower()

    # 移除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)

    # 提取最可能的关键词：寻找 yes / no / possibly / possible / maybe / probably 等
    if re.search(r'\byes\b', answer):
        return "Yes."
    elif re.search(r'\bno\b', answer):
        return "No."
    elif re.search(r'\b(possibly|possible|maybe|perhaps|probably)\b', answer):
        return "Possibly."
    else:
        return "Unknown"


def is_answer_correct(generated_answer):
    """
    判断生成的回答是否与预期答案相匹配。

    :param generated_answer: str, 生成的回答
    :param expected_answer: str, 预期答案
    :return: bool, 如果匹配成功返回True，否则返回False
    """

    expected_answer = 'image 2'
    # 清理生成的回答和预期答案：转换为小写，移除首尾空格
    cleaned_generated = generated_answer.strip().lower()
    cleaned_expected = expected_answer.strip().lower()

    if cleaned_generated == cleaned_expected:
        return 'image 2'
    else:
        return 'image 1'


def evaluate_model(model, val_loaders, image_processor, tokenizer,device, epoch, args, rank):
    model.eval()



    cls_nums_all = 0
    cls_acc_all = 0
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()


    for batch in tqdm(val_loaders, desc=f"[Rank {rank}] Epoch {epoch + 1}/{args.epochs}"):

        messages = batch['messages']

        answers = batch['answers'][0]


        res = model.module.chat(messages[0], tokenizer, image_processor, max_new_tokens=1024, do_sample=True, temperature=0.6)


        is_match = is_answer_correct(res)

        if is_match == answers:
            cls_acc_all += 1

        cls_nums_all += 1



    return cls_acc_all / cls_nums_all


# ------------------- 分布式初始化 -------------------
def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Initialized (world_size={world_size})")


def cleanup():
    dist.destroy_process_group()


def set_seed(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------- Dataloader -------------------
def collate_fn(batch, processor, tokenizer, device):
    pad_token_id = tokenizer.pad_token_id or 0
    pixel_values_list = [b['pixel_values'] for b in batch]
    input_ids_list = [b['input_ids'] for b in batch]
    image_flags_list = [b['image_flags'] for b in batch]
    labels_list = [b['labels'] for b in batch]

    max_len = max(len(ids) for ids in input_ids_list)
    bs = len(batch)

    padded_input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    padded_image_flags = torch.zeros((bs, max_len), dtype=torch.int)
    padded_labels = torch.full((bs, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)

    for i, (input_ids, image_flags, labels) in enumerate(zip(input_ids_list, image_flags_list, labels_list)):
        seq_len = len(input_ids)
        padded_input_ids[i, :seq_len] = input_ids
        padded_image_flags[i, :seq_len] = image_flags
        padded_labels[i, :seq_len] = labels
        attention_mask[i, :seq_len] = 1

    pixel_values = torch.stack(pixel_values_list) if all(
        pv.shape == pixel_values_list[0].shape for pv in pixel_values_list
    ) else pixel_values_list

    return {
        'pixel_values': pixel_values,
        'input_ids': padded_input_ids,
        'image_flags': padded_image_flags,
        'labels': padded_labels,
        'attention_mask': attention_mask
    }


def collate_test(batch):
    messages = [item['message'] for item in batch]
    answers = [item['answers'] for item in batch]

    return {
        'messages': messages,
        'answers': answers
    }


def create_data_loaders(train_dataset, val_datasets, batch_size, num_workers, rank, world_size, processor, tokenizer, device):
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, processor=processor, tokenizer=tokenizer, device=device)
    )

    val_sampler = DistributedSampler(val_datasets, num_replicas=world_size, rank=rank)
    val_loaders = DataLoader(
        val_datasets,
        batch_size=1,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate_test)
    )
    return train_loader, val_loaders


# ------------------- 主训练函数 -------------------
def train_model(rank, world_size, args):
    setup(rank, world_size)
    set_seed(args.seed, rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project=args.project_name, name=args.run_name)
        wandb.config.update(vars(args))

    # 加载模型和处理器
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = CLIPImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.train_json, "r") as f:
        train_data = json.load(f)
    with open(args.val_json, "r") as f:
        val_data = json.load(f)
    train_dataset = PedestrianMatchingDataset(data=train_data, image_processor=processor, tokenizer=tokenizer)
    val_datasets = PedestrianMatchingDataset_Test(data=val_data, image_processor=processor, tokenizer=tokenizer)

    model = DDP(model, device_ids=[rank])
    # device = 'cuda'

    train_loader, val_loaders = create_data_loaders(
        train_dataset, val_datasets, args.batch_size, 0, rank, world_size, processor, tokenizer, device
    )

    # ---------------- 优化器与学习率调度 ----------------
    base_lr = args.lr * args.lr_scale
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    if args.lr_scheduler == "cosine":
        lr_scheduler = get_scheduler(
            name="cosine", optimizer=optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif args.lr_scheduler == "linear":
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif args.lr_scheduler == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            anneal_strategy='cos',
            pct_start=args.warmup_ratio,
        )
    else:  # constant
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # ---------------- 训练循环 ----------------
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch + 1}/{args.epochs}"):
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            image_flags = batch['image_flags'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
                return_dict=True
            )

            llm_loss = outputs.loss

            llm_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += llm_loss.item()
            global_step += 1

            # wandb记录
            if rank == 0 and global_step % 10 == 0:
                wandb.log({
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "llm_loss": llm_loss.item()
                })

        avg_loss = total_loss / len(train_loader)

        acc = evaluate_model(model, val_loaders, processor, tokenizer, device, epoch, args, rank)

        if rank == 0:


            print(f"[Rank {rank}] Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss, "acc": acc,})
            # 保存检查点
            save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"[Rank {rank}] Saving checkpoint to {save_dir}")
            model.module.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"[Rank {rank}] Checkpoint saved successfully")

    if rank == 0:
        wandb.finish()
    cleanup()


# ------------------- 主入口 -------------------
def main():
    parser = argparse.ArgumentParser(description="Train AndesVL with LR optimization (torchrun compatible)")
    parser.add_argument("--dataset", type=str, default="DGM4")
    parser.add_argument("--train-json", type=str, default="top_data/train_top2_datas.json")
    parser.add_argument("--val-json", type=str, default="top_data/test_top2_data.json")
    parser.add_argument("--model-path", type=str, default="./AndesVL-1B-Instruct")
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine schedule")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "linear", "onecycle", "constant"])
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup step ratio")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Learning rate scale factor (for batch-size scaling)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--regular-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--save-steps", type=int, default=None, help="Save checkpoint every N steps (default: every epoch)")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--project-name", type=str, default="Person_top2")
    parser.add_argument("--output-dir", type=str, default="./Andes_log")
    parser.add_argument("--run-name", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()

    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    # formatted_now = '2025-02-25_10-34-13'
    args.output_dir = args.output_dir + '/' + formatted_now

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    train_model(rank, world_size, args)


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "3df743f5636d10739be1362c9033aa232f91df8a"
    os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
    main()

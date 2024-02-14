import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import wandb
import math
import copy
import os
import argparse
import PIL

import sys
sys.path.append(".")
sys.path.append("..")
from src.models.vision_transformer import vit_huge
from utils import evaluate, param_groups_lrd

from timm.data.mixup import Mixup
from timm.data import create_transform
# from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_

torch.cuda.manual_seed_all(0)
np.random.seed(0)

class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

parser = ArgumentParser(
    description="Train ViT on ImageNet.",
    fromfile_prefix_chars="@")

# Path arguments
parser.add_argument("--model_name", type=str, default="vit_s_16")
parser.add_argument("--save_dir", type=str, default="models")
parser.add_argument("--data_dir", type=str, default="data/imagenet")

# Training arguments
parser.add_argument("--num_workers", type=int, default=9)
parser.add_argument("--devices", type=int, nargs="+", default=[0])
parser.add_argument("--train_effective_batch_size", type=int, default=4096)
parser.add_argument("--max_batch_size_per_device", type=int, default=256)
parser.add_argument("--val_split_size", type=int, default=10000)
parser.add_argument("--val_effective_batch_size", type=int, default=4096)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--warmup_steps", type=int, default=10000)
parser.add_argument("--total_epochs", type=int, default=90)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--master_port", type=int, default=29500)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--pretrained_path", type=str, default='pre-training_weights/IN1K-vit.h.14-300e.pth.tar')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

# create_transform
parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')


args = parser.parse_args()

wandb_id_path = "{}/{}_wandb_id.txt".format(args.save_dir, args.model_name)
checkpoint_path = "{}/{}_checkpoint.pt".format(args.save_dir, args.model_name)
split_path = "{}/{}_split.npz".format(args.save_dir, args.model_name)
last_save_gpu_path = "{}/{}_last_save_gpu.txt".format(
    args.save_dir, args.model_name)
model_path = "{}/{}.pt".format(args.save_dir, args.model_name)
last_epoch_model_path = "{}/{}_last_epoch_model.pt".format(args.save_dir, args.model_name)

use_distributed = len(args.devices) > 1 and torch.cuda.is_available()

train_set_size = 1281167
train_split_size = train_set_size - args.val_split_size
train_split_per_device_size = (train_split_size + \
    (train_split_size % len(args.devices))) // len(args.devices)
train_batch_size = args.train_effective_batch_size // len(args.devices)
val_batch_size = args.val_effective_batch_size // len(args.devices)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def build_val_transform():
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(imagenet_mean, imagenet_std))
    return transforms.Compose(t)

def train(rank, world_size, train_set_indices, val_set_indices, last_save_gpu):
    if use_distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    device_id = args.devices[rank]
    device_string = "cuda:{}".format(device_id)
    device = torch.device(device_string if torch.cuda.is_available() else "cpu")

    if rank == 0 and not args.debug:
        wandb.login()

        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, mode="r") as f:
                wandb_id = f.read()
        else:
            wandb_id = wandb.util.generate_id()

            with open(wandb_id_path, mode="w") as f:
                f.write(wandb_id)

        wandb.init(
            project="ijepa",
            name=args.model_name,
            id=wandb_id,
            resume="allow",
            entity="common-corruptions")

    start_epoch = 0
    
    train_transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=imagenet_mean,
            std=imagenet_std,
        )

    val_transform = build_val_transform()

    original_train_set = datasets.ImageNet(
        root=args.data_dir, split="train",
        transform=train_transform)
    original_val_set = datasets.ImageNet(
        root=args.data_dir, split="train",
        transform=val_transform) 

    train_set = torch.utils.data.Subset(
        original_train_set, train_set_indices)
    val_set = torch.utils.data.Subset(
        original_val_set, val_set_indices)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank, 
        shuffle=True,
    ) if use_distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False, 
        drop_last=True,
    ) if use_distributed else None

    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=(None if use_distributed else True), 
        sampler=train_sampler, 
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_set_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=(None if use_distributed else False),
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    original_model = vit_huge(patch_size=14, drop_path_rate=0.2).to(device)
    model = original_model

    # Load pretrained models
    if args.finetune:
        # map_location = {"cuda:%d" % last_save_gpu: device_string}

        pretrained = torch.load(
            args.pretrained_path, map_location=device_string)

        pretrained_model = pretrained['target_encoder']
        new_pretrained_model = {key.replace('module.', ''): value for key, value in pretrained_model.items()}
        
        model.load_state_dict(new_pretrained_model, strict=False)
        trunc_normal_(model.fc.weight, std=2e-5)

    if use_distributed:
        model = DDP(model, device_ids=[device_id])
        model_without_ddp = model.module

    if args.smoothing > 0.:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list={'pos_embed', 'cls_token', 'dist_token'},
        layer_decay=args.layer_decay
    )
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate)
    
    schedulers = []

    if args.warmup_steps > 0:
        schedulers.append(
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=(1 / args.warmup_steps),
                total_iters=args.warmup_steps
            )
        )

    schedulers.append(
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(args.total_epochs - args.warmup_steps),
        )
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers,
        [args.warmup_steps],
    )

    if rank == 0:
        best_val_loss = np.inf
        best_model_state = None
 
    # Load checkpoint
    if os.path.exists(checkpoint_path) and last_save_gpu is not None:
        map_location = {"cuda:%d" % last_save_gpu: device_string}

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location)
        
        model.load_state_dict(checkpoint["model_state_dict"])

        if rank == 0:
            best_val_loss = checkpoint["best_val_loss"]
            best_model_state = checkpoint["best_model_state_dict"]
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]

        if rank == 0:    
            print("Resuming training from epoch {}...".format(start_epoch + 1))

    epoch_progress = range(start_epoch, args.total_epochs)

    if rank == 0:
        epoch_progress = tqdm(
            epoch_progress, total=args.total_epochs, initial=start_epoch, 
            position=0, desc="Epochs", dynamic_ncols=True)
        
    #mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=1000)

    # Training loop
    for epoch in epoch_progress:
        if use_distributed:
            train_sampler.set_epoch(epoch)

        epoch_train_xe_loss = torch.tensor(
            0.0, dtype=torch.float32, device=device)
        total_items = torch.tensor(0, dtype=torch.int, device=device)

        batch_progress = train_set_loader

        if rank == 0:
            batch_progress = tqdm(
                batch_progress, position=1, leave=False, desc="Batches",
                dynamic_ncols=True)

        model.train()

        for data in batch_progress:
            if mixup_fn is not None:
                inputs, labels = mixup_fn(data[0].to(device), data[1].to(device))
            else:
                inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()

            iters = math.ceil(len(inputs) / args.max_batch_size_per_device)

            for i in range(iters):
                start = i * args.max_batch_size_per_device
                end = (i + 1) * args.max_batch_size_per_device
                iter_inputs = inputs[start:end]
                iter_labels = labels[start:end]

                iter_outputs = model(iter_inputs)

                if len(labels) == 1:
                    iter_outputs = torch.unsqueeze(iter_outputs, 0)

                xe_loss = loss_fn(iter_outputs, iter_labels)
                loss = len(iter_outputs) * xe_loss / len(inputs)

                epoch_train_xe_loss += loss * len(inputs)

                loss.backward()

            total_items += len(inputs)

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)


            optimizer.step()

        scheduler.step()

        if use_distributed:
            dist.all_reduce(
                epoch_train_xe_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(
                total_items, op=dist.ReduceOp.SUM)
        
        epoch_train_xe_loss /= total_items

        # Evaluate on disjoint validation data selected from original train set
        model.eval()

        """
        modify the evaluate function to not take loss_fn as loss function but always use crossentropy,
        as implemented in the MAE finetune.
        """
        _, _, correct, total_epoch_val_xe_loss, total_items = evaluate(
            model,
            val_set_loader,
            show_progress=(rank == 0),
            device=device
        )
        
        if use_distributed:
            dist.all_reduce(
                correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(
                total_epoch_val_xe_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(
                total_items, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            epoch_val_xe_loss = total_epoch_val_xe_loss / total_items
            accuracy = correct / total_items * 100

            epoch_val_loss = epoch_val_xe_loss

            if epoch_val_loss < best_val_loss:
                best_model_state = copy.deepcopy(
                    model.module.state_dict() if use_distributed else\
                        original_model.state_dict())
                best_val_loss = epoch_val_loss

            if not args.debug:
                wandb.log({
                    "train_loss": epoch_train_xe_loss,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": accuracy,
                    "lr": scheduler.get_last_lr()[0],
                })

            # Save current checkpoint
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            if not args.debug:
                torch.save({
                    "wandb_id": wandb_id,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, checkpoint_path)
            else:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, checkpoint_path)

            if epoch == start_epoch:
                with open(last_save_gpu_path, mode="w") as f:
                    f.write(str(args.devices[0]))

    if rank == 0:
        torch.save(best_model_state, model_path)
        torch.save(original_model.state_dict(), last_epoch_model_path)
        
        if not args.debug:
            os.remove(wandb_id_path)
        os.remove(checkpoint_path)
        os.remove(split_path)
        os.remove(last_save_gpu_path)
        
        if not args.debug:
            wandb.finish()

    if use_distributed:
        dist.destroy_process_group()

def main():
    if os.path.exists(split_path):
        split = np.load(split_path)

        train_set_indices = split["train_set_indices"]
        val_set_indices = split["val_set_indices"]
    else:
        os.makedirs(args.save_dir)
        shuffled_indices = np.arange(train_set_size)
        np.random.shuffle(shuffled_indices)

        train_set_indices = shuffled_indices[args.val_split_size:]
        val_set_indices = shuffled_indices[:args.val_split_size]
        
        np.savez(
            split_path, train_set_indices=train_set_indices,
            val_set_indices=val_set_indices)
    
    last_save_gpu = None
    
    if os.path.exists(last_save_gpu_path):
        with open(last_save_gpu_path, mode="r") as f:
            last_save_gpu = int(f.read())

    if use_distributed:
        world_size = len(args.devices)

        mp.spawn(train,
            args=(
                world_size,
                train_set_indices,
                val_set_indices,
                last_save_gpu,
            ),
            nprocs=world_size,
            join=True)
    else:
        train(0, 1, train_set_indices, val_set_indices, last_save_gpu)

if __name__=="__main__":
    if use_distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.master_port)

    main()
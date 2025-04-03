"""
train_masam.py

Main training script for MASAM model using:
- CrossEntropy + Dice loss
- Warmup + cosine LR schedule
- Mixed precision + checkpointing
"""

import os
import sys
import math
import logging
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from src.utils import DiceLoss
from src.dataset import dataset_reader, RandomGenerator

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists('./training_log'):
        os.mkdir('./training_log')

    logging.basicConfig(filename='./training_log/' + args.output.split('/')[-1] + '_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = dataset_reader(base_dir=args.root_path, split="train", num_classes=args.num_classes,
                              transform=transforms.Compose([
                                  RandomGenerator(output_size=[args.img_size, args.img_size],
                                                  low_res=[low_res, low_res])]))
    logging.info("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(num_classes + 1)

    b_lr = base_lr / args.warmup_period if args.warmup else base_lr

    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = max_epoch * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, bbox_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['prompt']
            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)
            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)
            bbox_batch = bbox_batch.contiguous().view(-1, 4)

            low_res_label_batch = sampled_batch['low_res_label']

            image_batch, label_batch, bbox_batch = image_batch.cuda(), label_batch.cuda(), bbox_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size, bbox_batch)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(image_batch, multimask_output, args.img_size, bbox_batch)
                loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                shift_iter = iter_num - args.warmup_period if args.warmup else iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(f'iteration {iter_num} : loss {loss.item():.4f}, loss_ce {loss_ce.item():.4f}, loss_dice {loss_dice.item():.4f}')

        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logging.info(f"Model saved to {save_mode_path}")

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logging.info(f"Final model saved to {save_mode_path}")
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

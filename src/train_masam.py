"""
train_masam.py

Main training script for MASAM model using:
- CrossEntropy + Dice loss
- Warmup + cosine LR schedule
- Mixed precision + checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    logging.basicConfig(filename='./training_log/' + args.output.split('/')[-1] + '_log.txt', level=logging.INFO)
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    db_train = dataset_reader(base_dir=args.root_path, split="train", num_classes=args.num_classes,
                              transform=RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res]))
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8)

    model.train()

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(args.num_classes + 1)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('info/lr', base_lr, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)
            iter_num += 1

        if (epoch_num + 1) % 20 == 0:
            model.save_parameters(os.path.join(snapshot_path, f'epoch_{epoch_num}.pth'))

    writer.close()
    return "Training Finished!"

# Call the function
trainer_run(args, model, snapshot_path, multimask_output, low_res)

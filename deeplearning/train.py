#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

from __future__ import absolute_import, division, print_function

import os.path as osp

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from tensorboardX import SummaryWriter
# from torch.autograd import Variable
#from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm
import yaml
from models.deeplabv3 import DeepLabV3
from utils.loss import CrossEntropyLoss2d
import utils.transforms as t
from utils.dataset import GIANA

def get_lr_params(model, key):
    # For Dilated FCN
    if key == '1x':
        for m in model.named_modules():
            if 'layer' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == '10x':
        for m in model.named_modules():
            if 'aspp' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == '20x':
        for m in model.named_modules():
            if 'aspp' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter)**power
    optimizer.param_groups[0]['lr'] = new_lr
    optimizer.param_groups[1]['lr'] = 10 * new_lr
    optimizer.param_groups[2]['lr'] = 20 * new_lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size, ) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()

def load_data(args):

    normalize = t.Normalize(mean=[0.445, 0.287, 0.190], std=[0.31, 0.225, 0.168])
    im_transform = t.Compose([t.ToTensor(), normalize])

    # Use  the following code fo co_transformations e.g. random rotation or random flip etc.
    # co_transformer = cot.Compose([cot.RandomRotate(45)])
    dsetTrain = GIANA(args.imgdir, args.gtdir, input_size=(args.input_width, args.input_height) ,train=True, transform=im_transform, co_transform=None, target_transform=t.ToLabel(), train_split=0.8)
    train_data_loader = data.DataLoader(dsetTrain, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dsetVal = GIANA(args.imgdir, args.gtdir, train=False, transform=im_transform, co_transform=None,
                    target_transform=t.ToLabel())
    val_data_loader = data.DataLoader(dsetVal, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
    return train_data_loader, val_data_loader


#@click.command()
#@click.option('--config', '-c', type=str, required=True)
#@click.option('--cuda/--no-cuda', default=True)
def main():
    # Configuration
    CONFIG = Dict(yaml.load(open('/home/modulabs/soojin/2018_stepping/deeplearning/giana.yaml')))

    # CUDA check
    cuda = torch.cuda.is_available()
    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on', torch.cuda.get_device_name(current_device))

    # Dataset
    # dataset = CocoStuff10k(
    #     root=CONFIG.ROOT,
    #     split='train',
    #     image_size=513,
    #     crop_size=CONFIG.IMAGE.SIZE.TRAIN,
    #     scale=True,
    #     flip=True,
    # )

    normalize = t.Normalize(mean=[0.445, 0.287, 0.190], std=[0.31, 0.225, 0.168])
    im_transform = t.Compose([t.ToTensor(), normalize])

    # Use  the following code fo co_transformations e.g. random rotation or random flip etc.
    # co_transformer = cot.Compose([cot.RandomRotate(45)])
    print(CONFIG.ROOT)
    print(CONFIG.LABELS)
    dataset = GIANA(CONFIG.ROOT, CONFIG.LABELS, input_size=(CONFIG.IMAGE.SIZE.WIDTH, CONFIG.IMAGE.SIZE.HEIGHT) ,train=True, transform=im_transform, co_transform=None, target_transform=t.ToLabel(), train_split=0.8)
    #train_data_loader = data.DataLoader(dsetTrain, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #dsetVal = GIANA(args.imgdir, args.gtdir, train=False, transform=im_transform, co_transform=None,
                    #target_transform=t.ToLabel())
    #val_data_loader = data.DataLoader(dsetVal, batch_size=args.batch_size, shuffle=False,
                                      #num_workers=args.num_workers)


    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model
    model = DeepLabV3(n_classes=CONFIG.N_CLASSES, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18])
    #state_dict = torch.load(CONFIG.INIT_MODEL)
    #model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    model = nn.DataParallel(model)
    if cuda:
        model.cuda()

    # Optimizer
    optimizer = {
        'sgd':
            torch.optim.SGD(
                # cf lr_mult and decay_mult in train.prototxt
                params=[{
                    'params': get_lr_params(model.module, key='1x'),
                    'lr': CONFIG.LR,
                    'weight_decay': CONFIG.WEIGHT_DECAY
                }, {
                    'params': get_lr_params(model.module, key='10x'),
                    'lr': 10 * CONFIG.LR,
                    'weight_decay': CONFIG.WEIGHT_DECAY
                }, {
                    'params': get_lr_params(model.module, key='20x'),
                    'lr': 20 * CONFIG.LR,
                    'weight_decay': 0.0
                }]
            ),
    }.get(CONFIG.OPTIMIZER)

    # Loss definition
    criterion = CrossEntropyLoss2d(ignore_index=CONFIG.IGNORE_LABEL)
    if cuda:
        criterion.cuda()

    # TensorBoard Logger
    writer = SummaryWriter(CONFIG.LOG_DIR)
    #loss_meter = MovingAverageValueMeter(20)

    model.train()
    model.module.scale.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.ITER_MAX + 1),
        total=CONFIG.ITER_MAX,
        leave=False,
        dynamic_ncols=True,
    ):

        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            try:
                data, target = next(loader_iter)
            except:
                loader_iter = iter(loader)
                data, target = next(loader_iter)

            # Image
            data = data.cuda() if cuda else data
            #data = Variable(data)

            # Propagate forward
            outputs = model(data)

            # Loss
            loss = 0
            for output in outputs:
                # Resize target for {100%, 75%, 50%, Max} outputs
                target_ = resize_target(target, output.size(2))
                target_ = target_.cuda() if cuda else target_
                # target_ = Variable(target_)
                # Compute crossentropy loss
                loss += criterion(output, target_)

            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += loss.data[0]

       # loss_meter.add(iter_loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # TensorBoard
        if iteration % CONFIG.ITER_TF == 0:
            #writer.add_scalar('train_loss', loss_meter.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar('train_lr_group{}'.format(i), o['lr'], iteration)
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                writer.add_histogram(name, param, iteration, bins="auto")
                if param.requires_grad:
                    writer.add_histogram(name + '/grad', param.grad, iteration, bins="auto")

        # Save a model
        if iteration % CONFIG.ITER_SNAP == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.SAVE_DIR, 'checkpoint_{}.pth'.format(iteration)),
            )

        # Save a model
        if iteration % 100 == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.SAVE_DIR, 'checkpoint_current.pth'),
            )

    torch.save(
        model.module.state_dict(),
        osp.join(CONFIG.SAVE_DIR, 'checkpoint_final.pth'),
    )


if __name__ == '__main__':
    main()

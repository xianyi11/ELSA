# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from random import random
import torch
from AutoAugment import CIFAR10Policy

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.dataPath, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_cifar10_dataset(is_train,args):
    # prepare and pre-process dataset
    data_path = args.dataPath

    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # padding 0 and crop
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703233,0.24348505,0.26158768)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703233,0.24348505,0.26158768)),
        ])

    dataset = datasets.CIFAR10(root=data_path, train=is_train, download=True, transform=transform)

    return dataset

def build_cifar100_dataset(is_train,args):
    # prepare and pre-process dataset
    data_path = args.dataPath

    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # padding 0 and crop
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    dataset = datasets.CIFAR100(root=data_path, train=is_train, download=True, transform=transform)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            no_aug=args.no_aug,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
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
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def mixup_pretrain(images,ratio):
    image_len = images.shape[0]
    
    image = []
    image1 = []
    image2 = []
    # ratio = random()
    for i in range(image_len):
        idx1 = int(random()*image_len)
        idx2 = int(random()*image_len)
        while(idx1 == idx2):
            idx2 = int(random()*image_len)
        image1.append(images[idx1])
        image2.append(images[idx2])
        image.append(images[idx1]*ratio + images[idx2]*(1-ratio))
    image = torch.stack(image)
    image1 = torch.stack(image1)
    image2 = torch.stack(image2)

    return image,image1,image2
        
    
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
from fmix import sample_mask
import torch


def totensor():
    """An example of custom transform function."""
    return A.pytorch.ToTensor()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = {
        'target': target,
        'shuffled_target': shuffled_target,
        'lam': lam
    }

    return new_data, targets


def fmix(data, target, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    x1 = torch.from_numpy(mask).to(data.device) * data
    x2 = torch.from_numpy(1 - mask).to(data.device) * shuffled_data
    targets = {
        'target': target,
        'shuffled_target': shuffled_target,
        'lam': lam
    }

    return x1 + x2, targets


def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha) # no clipping to take a whole image
    new_data = data.clone()
    new_data = data * lam + shuffled_data * (1.0 - lam)
    targets = {
        'target': target,
        'shuffled_target': shuffled_target,
        'lam': lam
    }

    return new_data, targets
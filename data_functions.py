import cv2
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from utils import get_transforms


def datagenerator(cfg):
    """Generates data (images and targets) for train and test."""

    print("Getting the data...")
    assert cfg.train_size + cfg.val_size + cfg.test_size == 1, "the sum of the split size must be equal to 1."
    df = pd.read_csv(cfg.path_to_csv)
    data = df[df.source == 2020][["image_id", "label"]]
    if cfg.debug:
        data = data.sample(n=1000, random_state=cfg.seed).reset_index(drop=True)

    if not cfg.kfold:
        targets = list(data["label"])
        files = list(data["image_id"])

        # If test size is equal zero, we split the data only into train and validation parts,
        # otherwise we split it into train, validation and test parts.
        trainimgs, testimgs, traintargets, testtargets = train_test_split(files, targets, train_size=cfg.train_size,
                                                                          test_size=cfg.val_size + cfg.test_size,
                                                                          random_state=cfg.seed, stratify=targets)
    else:
        kf = KFold(n_splits=cfg.n_splits, shuffle=False)
        fold = 1
        for train_index, test_index in kf.split(data):
            if fold == cfg.fold:
                trainimgs = list(data.iloc[train_index]["image_id"])
                traintargets = list(data.iloc[train_index]["label"])
                testimgs = list(data.iloc[test_index]["image_id"])
                testtargets = list(data.loc[test_index]["label"])
                break
            fold += 1

    trainimgs += list(df[df.source == 2019].image_id)
    traintargets += list(df[df.source == 2019].label)
    cfg.scheduler_params["steps"] = len(trainimgs) * 110 / cfg.train_batchsize
    if cfg.test_size == 0:
        return trainimgs, traintargets, testimgs, testtargets

    valimgs, testimgs, valtargets, testtargets = train_test_split(testimgs, testtargets,
                                                                  train_size=cfg.val_size,
                                                                  test_size=cfg.test_size,
                                                                  random_state=cfg.seed,
                                                                  stratify=testtargets)
    return trainimgs, traintargets, valimgs, valtargets, testimgs, testtargets


class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""

    def __init__(self, cfg, images, targets, transforms):
        self.images = images  # List with images
        self.targets = targets  # List with targets
        self.transforms = transforms  # Transforms
        self.cfg = cfg  # Config

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.cfg.path_to_imgs, self.images[idx]))
        img = self.transforms(image=img)["image"]
        img = torch.FloatTensor(img)
        target = torch.LongTensor([int(self.targets[idx])])
        return img, target

    def __len__(self):
        return len(self.targets)


def get_loaders(cfg):
    """Getting dataloaders for train, validation (and test, if needed)."""
    trainforms, testforms = get_transforms(cfg)

    # If test size is equal zero, we create the loaders only for train and validation parts,
    # otherwise we create the loaders for train, validation and test parts.
    if cfg.test_size != 0.0:
        trainimgs, traintargets, valimgs, valtargets, testimgs, testtargets = datagenerator(cfg)
        traindataset = CassavaDataset(cfg, trainimgs, traintargets, trainforms)
        valdataset = CassavaDataset(cfg, valimgs, valtargets, testforms)
        testdataset = CassavaDataset(cfg, testimgs, testtargets, testforms)
        trainloader = torch.utils.data.DataLoader(traindataset,
                                                  shuffle=cfg.train_shuffle,
                                                  batch_size=cfg.train_batchsize,
                                                  pin_memory=False,
                                                  num_workers=cfg.num_workers,
                                                  persistent_workers=True)
        valloader = torch.utils.data.DataLoader(valdataset,
                                                shuffle=cfg.val_shuffle,
                                                batch_size=cfg.val_batchsize,
                                                pin_memory=False,
                                                num_workers=cfg.num_workers,
                                                persistent_workers=True)
        testloader = torch.utils.data.DataLoader(testdataset,
                                                 shuffle=cfg.test_shuffle,
                                                 batch_size=cfg.test_batchsize,
                                                 pin_memory=False,
                                                 num_workers=cfg.num_workers,
                                                 persistent_workers=True)
        return trainloader, valloader, testloader

    else:
        trainimgs, traintargets, valimgs, valtargets = datagenerator(cfg)
        traindataset = CassavaDataset(cfg, trainimgs, traintargets, trainforms)
        valdataset = CassavaDataset(cfg, valimgs, valtargets, testforms)
        trainloader = torch.utils.data.DataLoader(traindataset,
                                                  shuffle=cfg.train_shuffle,
                                                  batch_size=cfg.train_batchsize,
                                                  pin_memory=False,
                                                  num_workers=cfg.num_workers,
                                                  persistent_workers=True)
        valloader = torch.utils.data.DataLoader(valdataset,
                                                shuffle=cfg.val_shuffle,
                                                batch_size=cfg.val_batchsize,
                                                pin_memory=False,
                                                num_workers=cfg.num_workers,
                                                persistent_workers=True)
        return trainloader, valloader

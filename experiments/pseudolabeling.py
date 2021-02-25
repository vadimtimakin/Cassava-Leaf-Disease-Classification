import torch
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from config import Cfg
import os
import timm
import torch.nn as nn
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensor
import warnings
from torchvision import models
warnings.filterwarnings("ignore")


def get_models(cfg):
    """Get PyTorch model."""
    models = list()
    for i in range(len(cfg.modelnames)):
        if cfg.modelnames[i].startswith("/timm/"):
            model = timm.create_model(cfg.modelnames[i][6:], pretrained=False)
        elif cfg.modelnames[i].startswith("/torch/"):
            model = getattr(models, cfg.modelnames[i][7:])(pretrained=False)

        lastlayer = list(model._modules)[-1]
        try:
            setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer).in_features,
                                                    out_features=cfg.NUMCLASSES, bias=True))
        except torch.nn.modules.module.ModuleAttributeError:
            setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer)[1].in_features,
                                                out_features=cfg.NUMCLASSES, bias=True))/home/toefl/K/cassava/mendeley
        model = model.to(cfg.device).eval()
        models.append(model)
    return models


class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""

    def __init__(self, cfg, images):
        self.images = images  # List with images
        self.cfg = cfg  # Config
        self.transforms = A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            A.pytorch.ToTensor(),
        ])

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.cfg.pathtoimgs, self.images[idx]))
        return torch.FloatTensor(self.transforms(image=img)["image"]), self.images[idx]

    def __len__(self):
        return len(self.images)

cfg = Cfg

cfg.chks = [
    "./fold1.pt",
    "./fold2.pt",
    "./fold3.pt",
    "./fold4.pt",
    "./fold5.pt"
]  # Path to model checkpoint (weights)
cfg.modelnames = [
    "/timm/swsl_resnext101_32x8d",
    "/timm/swsl_resnext101_32x8d",
    "/timm/swsl_resnext101_32x8d",
    "/timm/swsl_resnext101_32x8d",
    "/timm/swsl_resnext101_32x8d",
]  # PyTorch model

df = pd.read_csv("./mendeley_leaf_data.csv")
cfg.pathtoimgs = "./mendeley_leaf_data"
print(df.head())
df["confidence"] = [0 for i in range(len(df))]
df["predict"] = [0 for i in range(len(df))]
df["label"] = df["label"].map({"healthy":4, "diseased": -1})
print(df.head())

print(len(df[df["predict"] != 0]))
models = get_models(cfg) 
imgs = list(df["image_id"])
dataset = CassavaDataset(cfg, imgs)
dataloader = torch.utils.data.DataLoader(dataset,
                        shuffle=False,
                        batch_size=1,
                        pin_memory=False,
                        num_workers=cfg.numworkers,
                        persistent_workers=True)

models = get_models(cfg)
for img, name in tqdm(dataloader):
    outputs = None
    for model in models:
        preds = model(img.to(cfg.device))[0]
        if outputs is not None:
            outputs += preds / 5
        else:
            outputs = preds / 5
    df["confidence"][df['image_id'] == name[0]] = float(max(outputs))
    df["predict"][df['image_id'] == name[0]] = int(np.argmax(outputs.cpu().detach().numpy()))


print(df.head(10))
df.to_csv("PATH_TO_SAVE")



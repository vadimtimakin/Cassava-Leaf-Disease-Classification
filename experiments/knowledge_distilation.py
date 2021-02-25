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
warnings.filterwarnings("ignore")


def get_model(cfg):
    """Get PyTorch model."""
#     model = getattr(models, cfg.modelname)(pretrained=False)
    if cfg.modelname.startswith("/timm/"):
        model = timm.create_model(cfg.modelname[6:], pretrained=False)
    elif cfg.modelname.startswith("/torch/"):
        model = getattr(models, cfg.modelname[7:])(pretrained=False)
    else:
        raise RuntimeError("Unknown model source. Use /timm/ or /torch/.")
    # Changing the last layer according the number of classes
    lastlayer = list(model._modules)[-1]
    try:
        setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer).in_features,
                                            out_features=cfg.NUMCLASSES, bias=True))
    except torch.nn.modules.module.ModuleAttributeError:
        setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer)[1].in_features,
                                            out_features=cfg.NUMCLASSES, bias=True))
    cp = torch.load(cfg.chk)
    if 'model' in cp:
        model.load_state_dict(cp['model'])
    else:
        model.load_state_dict(cp)
    return model.to(cfg.device)


class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""

    def __init__(self, cfg, images):
        self.images = images  # List with images
        self.cfg = cfg  # Config
        self.transforms = A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            A.pytorch.ToTensor(),
        ]
        )

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.cfg.pathtoimgs, self.images[idx]))
        return torch.FloatTensor(self.transforms(image=img)["image"]), self.images[idx]

    def __len__(self):
        return len(self.images)

cfg = Cfg

df = pd.read_csv(cfg.pathtocsv)
print(df.head())
df["confidence"] = [0 for i in range(len(df))]
df["predict"] = [0 for i in range(len(df))]
df["fold"] = [0 for i in range(len(df))]
print(df.head())

kf5 = KFold(n_splits=5, shuffle=False)
fold = 1
for train_index, test_index in kf5.split(df):
    print(len(df[df["predict"] != 0]))
    cfg.chk = f"./fold{fold}.pt"
    model = get_model(cfg)
    model.eval()
    imgs = list(df.iloc[test_index]["image_id"])
    dataset = CassavaDataset(cfg, imgs)
    dataloader = torch.utils.data.DataLoader(dataset,
                            shuffle=False,
                            batch_size=1,
                            pin_memory=False,
                            num_workers=cfg.numworkers,
                            persistent_workers=True)
    for img, name in tqdm(dataloader):
        outputs = model(img.to(cfg.device))[0]
        df["confidence"][df['image_id'] == name[0]] = float(max(outputs))
        df["predict"][df['image_id'] == name[0]] = int(np.argmax(outputs.cpu().detach().numpy()))
        df["fold"][df['image_id'] == name[0]] = fold
        
    fold += 1

    print(df.head(10))

df.to_csv("PATH_TO_SAVE")



import random
import os
import numpy as np
import torch
import torchvision.models as models
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn.modules.module import ModuleAttributeError
import timm
import torch.nn as nn
import albumentations as A
# import wandb

from config import Cfg
from custom_functions.augmentations import totensor
from custom_functions.lossfn import LabelSmoothingLoss
from custom_functions.scheduler import CosineBatchDecayScheduler, GradualWarmupSchedulerV2
from custom_functions.optimizer import Ranger


def fullseed(seed=0xFACED):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model(cfg):
    """Get PyTorch model."""
    if cfg.chk:  # Loading model from the checkpoint
        print("Model:", cfg.model_name)
        if cfg.model_name.startswith("/timm/"):
            model = timm.create_model(cfg.model_name[6:], pretrained=False)
        elif cfg.model_name.startswith("/torch/"):
            model = getattr(models, cfg.model_name[7:])(pretrained=False)
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
        epoch, trainloss, valloss, metric, lr, stopflag = None, None, None, None, None, None
        if 'model' in cp:
            model.load_state_dict(cp['model'])
        else:
            model.load_state_dict(cp)
        if 'epoch' in cp:
            epoch = int(cp['epoch'])
            cfg.epoch = epoch
            cfg.warmup_epochs = max(1, cfg.warmup_epochs - epoch)
        if 'trainloss' in cp:
            trainloss = cp['trainloss']
        if 'valloss' in cp:
            valloss = cp['valloss']
        if 'metric' in cp:
            metric = cp['metric']
        if 'optimizer' in cp:
            cfg.optim_dict = cp['optimizer']
            lr = cp['optimizer']["param_groups"][0]['lr']
        if 'stopflag' in cp:
            stopflag = cp['stopflag']
            cfg.stopflag = stopflag
        if 'scheduler' in cp:
            cfg.scheduler_state = cp['scheduler']
        if 'size' in cp:
            cfg.start_size = cp['size']
        print("Uploading model from the checkpoint...",
              "\nEpoch:", epoch,
              "\nTrain Loss:", trainloss,
              "\nVal Loss:", valloss,
              "\nMetrics:", metric,
              "\nlr:", lr,
              "\nstopflag:", stopflag)
    else:  # Creating a new model
        print("Model:", cfg.model_name)
        if cfg.model_name.startswith("/timm/"):
            model = timm.create_model(cfg.model_name[6:], pretrained=cfg.pretrained)
        elif cfg.model_name.startswith("/torch/"):
            model = getattr(models, cfg.model_name[7:])(pretrained=cfg.pretrained)
        else:
            raise RuntimeError("Unknown model source. Use /timm/ or /torch/.")
        # Changing the last layer according the number of classes
        lastlayer = list(model._modules)[-1]
        try:
            setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer).in_features,
                                                out_features=cfg.NUMCLASSES, bias=True))
        except ModuleAttributeError:
            setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer)[1].in_features,
                                                out_features=cfg.NUMCLASSES, bias=True))
    return model.to(cfg.device)


def get_optimizer(model, cfg):
    "Get PyTorch optimizer."
    optimizer = globals()[cfg.optimizer[8:]](model.parameters(), **cfg.optimizer_params) \
        if cfg.optimizer.startswith("/custom/") \
        else getattr(optim, cfg.optimizer)(model.parameters(), **cfg.optimizer_params)
    if cfg.optim_dict:
        optimizer.load_state_dict(cfg.optim_dict)
    return optimizer


def get_scheduler(optimizer, cfg):
    """Get PyTorch scheduler."""
    afterone = globals()[cfg.scheduler[8:]](optimizer, **cfg.scheduler_params) \
        if cfg.scheduler.startswith("/custom/") \
        else getattr(optim.lr_scheduler, cfg.scheduler)(optimizer, **cfg.scheduler_params)
    scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=100, total_epoch=cfg.warmup_epochs, after_scheduler=afterone)
    if cfg.scheduler_state:
        scheduler.load_state_dict(cfg.scheduler_state)
    return scheduler


def get_lossfn(cfg):
    """Get PyTorch loss function."""
    return globals()[cfg.lossfn[8:]](**cfg.lossfn_params) \
        if cfg.lossfn.startswith("/custom/") \
        else getattr(nn, cfg.lossfn)(**cfg.lossfn_params)


def get_transforms(cfg):
    """Get train and test augmentations."""
    pretransforms = [globals()[item["name"][8:]](**item["params"]) if item["name"].startswith("/custom/")
                     else getattr(A, item["name"])(**item["params"]) for item in cfg.pretransforms]
    augmentations = [globals()[item["name"][8:]](**item["params"]) if item["name"].startswith("/custom/")
                     else getattr(A, item["name"])(**item["params"]) for item in cfg.augmentations]
    posttransforms = [globals()[item["name"][8:]](**item["params"]) if item["name"].startswith("/custom/")
                      else getattr(A, item["name"])(**item["params"]) for item in cfg.posttransforms]
    train = A.Compose(pretransforms + augmentations + posttransforms)
    test = A.Compose(pretransforms + posttransforms)
    return train, test


def savemodel(model, epoch, trainloss, valloss, metric, optimizer, stopflag, name, scheduler, size):
    """Saves PyTorch model."""
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'trainloss': trainloss,
        'valloss': valloss,
        'metric': metric,
        'optimizer': optimizer.state_dict(),
        'stopflag': stopflag,
        'scheduler': scheduler.state_dict(),
        'size': size,
    }, os.path.join(Cfg.path, name))


def drawplot(trainlosses, vallosses, metrics, lrs):
    """Draws plots of loss changes and test metric changes."""
    # Learning rate changes
    plt.plot(range(len(lrs)), lrs, label='Learning rate')
    plt.legend()
    plt.title("Learning rate changes")
    plt.show()
    # Validation and train loss changes
    plt.plot(range(len(trainlosses)), trainlosses, label='Train Loss')
    plt.plot(range(len(vallosses)), vallosses, label='Val Loss')
    plt.legend()
    plt.title("Validation and train loss changes")
    plt.show()
    # Test metrics changes
    plt.plot(range(len(metrics)), metrics, label='Metrics')
    plt.legend()
    plt.title("Test metrics changes")
    plt.show()


def printreport(t, trainloss, valloss, metric, record):
    """Prints epoch's report."""
    print(f'Time: {t} s')
    print(f'Train Loss: {trainloss:.4f}')
    print(f'Val Loss: {valloss:.4f}')
    print(f'Metrics: {metric:.4f}')
    print(f'Current best metrics: {record:.4f}')


def savelog(path, epoch, trainloss, valloss, metric):
    """Saves the epoch's log."""
    with open(path, "a") as file:
        file.write("epoch: " + str(epoch) + " trainloss: " + str(
            round(trainloss, 5)) + " valloss: " + str(
                round(valloss, 5)) + " metrics: " + str(round(metric, 5)) + "\n")
    # wandb.log({"epoch": str(epoch),
    #           "trainloss": str(round(trainloss, 5)),
    #           "valloss": str(round(valloss, 5)),
    #           "metrics": str(round(metric, 5))})
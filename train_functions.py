import gc
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.cuda import amp
import torch
# import wandb

from data_functions import *
from utils import *
from custom_functions.augmentations import cutmix, mixup, fmix
from custom_functions.scheduler import GradualWarmupSchedulerV2


def train(cfg, model, trainloader, optimizer, lossfn, scheduler, epoch, size, scaler):
    """Train loop."""
    print("Training")
    model.train()
    # Freezing BatchNorm layers
    for name, child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
    totalloss = 0.0

    for step, batch in enumerate(tqdm(trainloader)):
        inputs, labels = batch
        labels = labels.squeeze(1).to(cfg.device)

        # Augmentations
        p = random.uniform(0, 1)
        if p < 0.25:
            inputs, labels = fmix(inputs, labels, 3.0, 1.0, (size, size))
        elif 0.25 <= p < 0.5:
            inputs, labels = cutmix(inputs, labels, alpha=1.0)
        elif 0.5 <= p < 0.75:
            inputs, labels = mixup(inputs, labels, alpha=1.0)

        if not cfg.gradient_accumulation:
            optimizer.zero_grad()

        if cfg.mixed_precision:
            with amp.autocast():
                outputs = model(inputs.to(cfg.device).float())
                if p < 0.75:
                    loss = lossfn(outputs, labels['target']) * labels['lam'] + lossfn(outputs,
                                                                                      labels['shuffled_target']) * (
                                       1.0 - labels['lam'])
                else:
                    loss = lossfn(outputs, labels)
                if cfg.gradient_accumulation:
                    loss = loss / cfg.iters_to_accumulate
        else:
            outputs = model(inputs.to(cfg.device).float())
            if p < 0.75:
                loss = lossfn(outputs, labels['target']) * labels['lam'] + lossfn(outputs,
                                                                                  labels['shuffled_target']) * (
                                   1.0 - labels['lam'])
            else:
                loss = lossfn(outputs, labels)

        totalloss += loss.item()
        if cfg.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if cfg.gradient_accumulation:
            if (step + 1) % cfg.iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        elif cfg.mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if epoch >= cfg.warmup_epochs + 1:
            scheduler.step()

    if epoch < cfg.warmup_epochs + 1:
        scheduler.step()

    print("Learning rate:", optimizer.param_groups[0]['lr'])
    return totalloss / len(trainloader)


def validation(model, valloader, lossfn, cfg):
    """Validation loop."""
    print("Validating")
    model.eval()
    totalloss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(valloader):
            inputs, labels = batch
            labels = labels.squeeze(1).to(cfg.device)
            outputs = model(inputs.to(cfg.device))
            for idx in np.argmax(outputs.cpu(), axis=1):
                preds.append([1 if idx == i else 0 for i in range(5)])
            for j in labels:
                targets.append([1 if i == j else 0 for i in range(5)])
            loss = lossfn(outputs, labels)
            totalloss += loss.item()

    score = f1_score(targets, preds, average='weighted')
    return totalloss / len(valloader), score


def test(cfg, model, testloader, lossfn):
    """Testing loop."""
    print("Testing")
    model.eval()
    totalloss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(testloader):
            inputs, labels = batch
            labels = labels.squeeze(1).to(cfg.device)
            outputs = model(inputs.to(cfg.device))
            for idx in np.argmax(outputs.cpu(), axis=1):
                preds.append([1 if idx == i else 0 for i in range(5)])
            for j in labels:
                targets.append([1 if i == j else 0 for i in range(5)])
            loss = lossfn(outputs, labels)
            totalloss += loss.item()

    score = f1_score(targets, preds, average='weighted')
    print("Test Loss:", totalloss / len(testloader),
          "\nTest metrics:", score)


def run(cfg):
    """Main function."""

    # Logging
    # wandb.init(project=cfg.experiment_name)

    # Creating working directory
    if not os.path.exists(cfg.path):
        os.makedirs(cfg.path)

    # Getting the objects
    torch.cuda.empty_cache()
    if cfg.test_size != 0.0:
        trainloader, valloader, testloader = get_loaders(cfg)
    else:
        trainloader, valloader = get_loaders(cfg)
    model = get_model(cfg)
    optimizer = get_optimizer(model, cfg)
    if cfg.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None
    scheduler = get_scheduler(optimizer, cfg)
    lossfn = get_lossfn(cfg)

    # Initializing metrics
    trainlosses, vallosses, metrics, lrs = [], [], [], []
    record = 0
    stopflag = cfg.stopflag if cfg.stopflag else 0
    print('Testing "' + cfg.experiment_name + '" approach.')
    if cfg.log:
        with open(os.path.join(cfg.path, cfg.log), "w") as file:
            file.write('Testing "' + cfg.experiment_name + '" approach.\n')

    # Training
    print("Have a nice training!")
    augs = cfg.augmentations
    size = cfg.start_size  # Image size (Using progressive image size)
    for epoch in range(cfg.epoch + 1, cfg.num_epochs + 1):
        print("\nEpoch:", epoch)

        if size < cfg.final_size and epoch > cfg.warmup_epochs:
            size += cfg.size_step
        if epoch < cfg.warmup_epochs + 1:
            cfg.augmentations = [
                dict(
                    name="HorizontalFlip",
                    params=dict(
                        always_apply=False,
                        p=0.5,
                    )
                )
            ]
        else:
            cfg.augmentations = augs
        cfg.pretransforms = [
            dict(
                name="Resize",
                params=dict(
                    height=size,
                    width=size,
                    p=1.0,
                )
            ),
        ]
        print("Image size:", size)

        if cfg.test_size != 0.0:
            trainloader, valloader, testloader = get_loaders(cfg)
        else:
            trainloader, valloader = get_loaders(cfg)
        start_time = time.time()

        trainloss = train(cfg, model, trainloader, optimizer, lossfn, scheduler, epoch, size, scaler)
        valloss, metric = validation(model, valloader, lossfn, Cfg)
        trainlosses.append(trainloss)
        vallosses.append(valloss)
        metrics.append(metric)
        lrs.append(optimizer.param_groups[0]['lr'])

        if metric > record:
            stopflag = 0
            record = metric
            savemodel(model, epoch, trainloss, valloss, metric,
                      optimizer, stopflag, os.path.join(cfg.path, 'thebest.pt'), scheduler, size)
            print('New record!')
        else:
            stopflag += 1
        if epoch % cfg.savestep == 0:
            savemodel(model, epoch, trainloss, valloss, metric,
                      optimizer, stopflag, os.path.join(cfg.path, f'{epoch}epoch.pt'), scheduler, size)
        t = int(time.time() - start_time)
        printreport(t, trainloss, valloss, metric, record)

        # Saving to the log
        if cfg.log:
            savelog(os.path.join(cfg.path, cfg.log), epoch, trainloss, valloss, metric)

        torch.cuda.empty_cache()
        gc.collect()

        # Early stopping
        if stopflag == cfg.early_stopping:
            print("Training has been interrupted because of early stopping.")
            break

    # Test
    if cfg.test_size != 0.0:
        test(cfg, model, testloader, lossfn)

    # Verbose
    if cfg.verbose:
        drawplot(trainlosses, vallosses, metrics, lrs)

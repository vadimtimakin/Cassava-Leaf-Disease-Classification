from warmup_scheduler import GradualWarmupScheduler
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineBatchDecayScheduler(_LRScheduler):
    """
    Custom scheduler with calculating learning rate according batchsize
    based on Cosine Decay scheduler. Designed to use scheduler.step() every batch.
    """

    def __init__(self, optimizer, steps, epochs, batchsize=128, decay=128, startepoch=1, minlr=1e-8, last_epoch=-1):
        """
        Args:
            optimizer (torch.optim.Optimizer): PyTorch Optimizer
            steps (int): total number of steps
            epochs (int): total number of epochs
            batchsize (int): current training batchsize. Default: 128
            decay (int): batchsize based on which the learning rate will be calculated. Default: 128
            startepoch (int): number of epoch when the scheduler turns on. Default: 1
            minlr (float): the lower threshold of learning rate. Default: 1e-8
            last_epoch (int): The index of last epoch. Default: -1.
        """
        decay = decay * math.sqrt(batchsize)
        self.stepsize = batchsize / decay
        self.startstep = steps / epochs * (startepoch - 1) * self.stepsize
        self.minlr = minlr
        self.steps = steps
        self.stepnum = 0
        super(CosineBatchDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Formula for calculating the learning rate."""
        self.stepnum += self.stepsize
        if self.stepnum < self.startstep:
            return [baselr for baselr in self.base_lrs]
        return [max(self.minlr, 1/2 * (1 + math.cos(self.stepnum * math.pi / self.steps)) * self.optimizer.param_groups[0]['lr']) for t in range(len(self.base_lrs))]


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]
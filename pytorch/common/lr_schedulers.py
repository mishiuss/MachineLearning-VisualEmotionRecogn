from pytorch.common.abstract.abstract_lr_scheduler import AbstractLRScheduler
import math


class ApplyLR():
    def __init__(self, scale_lr=[1,1], scale_lr_fc=[1,1]):
        self.scale_lr = scale_lr
        self.scale_lr_fc = scale_lr_fc

    def apply(self, optimizer, lr, batch_idx):
        # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
        if (batch_idx < self.scale_lr[1]):
            damping = 0.5 * (1. + math.cos(math.pi * (batch_idx / float(self.scale_lr[1]))))
            if self.scale_lr[0] < 1.:
                lr *= 1. - (1. - self.scale_lr[0]) * damping
            else:
                lr *= 1. + (self.scale_lr[0] - 1.) * damping
        try:
            for param_group in optimizer.param_groups:
                if 'fc' in param_group['name']:
                    param_group['lr'] = lr
                    if (batch_idx < self.scale_lr_fc[1]):
                        damping = 0.5 * (1. + math.cos(math.pi * (batch_idx / float(self.scale_lr_fc[1]))))
                        if self.scale_lr_fc[0] > 1.:
                            param_group['lr'] *= 1. + (self.scale_lr_fc[0] - 1.) * damping
                        else:
                            param_group['lr'] *= 1. - (1. - self.scale_lr_fc[0]) * damping
                else:
                    param_group['lr'] = lr
        except:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


class SGDR_scheduler(AbstractLRScheduler):
    def __init__(self, optimizer, lr_start, lr_end, lr_period, scale_lr=[1,1], scale_lr_fc=[1,1]):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_period = lr_period
        self.lr_curr = lr_start
        self.apply_lr = ApplyLR(scale_lr, scale_lr_fc)
        self.batch_idx = 0

    def step(self):
        # returns normalised anytime sgdr schedule given period and batch_idx
        # best performing settings reported in paper are T_0 = 10, T_mult=2
        # so always use T_mult=2
        while self.batch_idx / float(self.lr_period) > 1.:
            self.batch_idx = self.batch_idx - self.lr_period
            self.lr_period *= 2.
            self.lr_start *= 0.2
            self.lr_end *= 0.2

        radians = math.pi * (self.batch_idx / float(self.lr_period))
        self.lr_curr = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (1. + math.cos(radians))
        self.apply_lr.apply(self.optimizer, self.lr_curr, self.batch_idx)
        self.batch_idx += 1

    def get_lr(self):
        return [self.lr_curr]


class LRFinder_scheduler():
    def __init__(self, optimizer, lr_start, lr_end, lr_period, use_linear_decay=True, scale_lr=[1,1], scale_lr_fc=[1,1]):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_period = lr_period
        self.lr_curr = lr_start
        self.use_linear_decay = use_linear_decay
        if self.use_linear_decay:
            self.k = (self.lr_end - self.lr_start) / self.lr_period
        else:
            self.k = (self.lr_end / self.lr_start) ** (1 / self.lr_period)
        self.apply_lr = ApplyLR(scale_lr, scale_lr_fc)
        self.batch_idx = 0

    def step(self):
        if self.batch_idx < self.lr_period:
            if self.use_linear_decay:
                self.lr_curr -= self.k
            else:
                self.lr_curr *= self.k
            self.apply_lr.apply(self.optimizer, self.lr_curr, self.batch_idx)
        self.batch_idx += 1

    def get_lr(self):
        return [self.lr_curr]


class OneCyclePolicy_scheduler(AbstractLRScheduler):
    def __init__(self, optimizer, lr_max, lr_period, use_linear_decay=True, scale_lr=[1,1], scale_lr_fc=[1,1]):
        self.optimizer = optimizer
        self.lr_start = lr_max / 25.
        self.lr_end = lr_max / 100.
        self.lr_period_1 = lr_period // 4
        self.lr_period_2 = self.lr_period_1 + lr_period
        self.lr_period_total = self.lr_period_2 + lr_period * 0.3
        self.lr_curr = self.lr_start
        self.use_linear_decay = use_linear_decay
        if self.use_linear_decay:
            self.k1 = (lr_max - self.lr_start) / self.lr_period_1
            self.k2 = (lr_max - self.lr_start) / (self.lr_period_2 - self.lr_period_1)
            self.k3 = (self.lr_end - self.lr_start) / (self.lr_period_total - self.lr_period_2)
        else:
            self.k1 = (lr_max / self.lr_start) ** (1 / self.lr_period_1)
            self.k2 = (lr_max / self.lr_start) ** (1 / (self.lr_period_2 - self.lr_period_1))
            self.k3 = (self.lr_end / self.lr_start) ** (1 / (self.lr_period_total - self.lr_period_2))
        self.apply_lr = ApplyLR(scale_lr, scale_lr_fc)
        self.batch_idx = 0

    def step(self):
        if self.use_linear_decay:
            if self.batch_idx < self.lr_period_1:
                self.lr_curr += self.k1
            elif self.batch_idx < self.lr_period_2:
                self.lr_curr -= self.k2
            elif self.batch_idx < self.lr_period_total:
                self.lr_curr += self.k3
        else:
            if self.batch_idx < self.lr_period_1:
                self.lr_curr *= self.k1
            elif self.batch_idx < self.lr_period_2:
                self.lr_curr /= self.k2
            elif self.batch_idx < self.lr_period_total:
                self.lr_curr *= self.k3

        if self.batch_idx < self.lr_period_total:
            self.apply_lr.apply(self.optimizer, self.lr_curr, self.batch_idx)

        self.batch_idx += 1

    def get_lr(self):
        return [self.lr_curr]


class MultiCyclePolicy_scheduler(AbstractLRScheduler):
    def __init__(self, optimizer, lr_max, lr_period, use_linear_decay=True, scale_lr=[1,1], scale_lr_fc=[1,1]):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_period = lr_period
        self.use_linear_decay = use_linear_decay
        self.apply_lr = ApplyLR(scale_lr, scale_lr_fc)
        self.batch_idx = 0
        self.lr_curr = self.lr_max / 10.
        self.create_cycle()

    def create_cycle(self):
        self.lr_start = self.lr_curr
        self.lr_period_1 = self.lr_period // 4
        self.lr_period_2 = self.lr_period_1 + self.lr_period
        if self.use_linear_decay:
            self.k1 = (self.lr_max - self.lr_start) / self.lr_period_1
            self.k2 = (self.lr_max - self.lr_start) / (self.lr_period_2 - self.lr_period_1)
        else:
            self.k1 = (self.lr_max / self.lr_start) ** (1 / self.lr_period_1)
            self.k2 = (self.lr_max / self.lr_start) ** (1 / (self.lr_period_2 - self.lr_period_1))

    def update_param(self):
        self.lr_max /= 3.
        self.lr_period /= 1.25
        self.create_cycle()
        self.lr_period_2 *= 1.1
        self.batch_idx = 0

    def step(self):
        if self.use_linear_decay:
            if self.batch_idx < self.lr_period_1:
                self.lr_curr += self.k1
            elif self.batch_idx < self.lr_period_2:
                self.lr_curr -= self.k2
            else:
                self.update_param()
        else:
            if self.batch_idx < self.lr_period_1:
                self.lr_curr *= self.k1
            elif self.batch_idx < self.lr_period_2:
                self.lr_curr /= self.k2
            else:
                self.update_param()

        self.lr_curr = max(0, self.lr_curr)
        self.apply_lr.apply(self.optimizer, self.lr_curr, self.batch_idx)
        self.batch_idx += 1

    def get_lr(self):
        return [self.lr_curr]
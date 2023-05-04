import warnings
import torch

class ReduceLROnPlateau():
    def init(self, model, curmonitor=torch.Tensor([float("inf")]), factor=0.1, patience=10, mode='min',
    min_delta=1e-4, cooldown=0, min_lr=0, verbose=1,**kwargs):
        self.curmonitor = curmonitor
        if factor > 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor > 1.0.')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.monitor_op = None
        self._reset()

    def _reset(self):
        if self.mode == 'min':
            self.monitor_op = lambda a, b: torch.less(a, b - self.min_delta)
            self.best = torch.Tensor([float("inf")])
        else:
            self.monitor_op = lambda a, b: torch.greater(a, b + self.min_delta)
            self.best = torch.Tensor([-float("inf")])
        self.cooldown_counter = 0
        self.wait = 0

    def update_monitor(self, curmonitor):
        self.curmonitor = curmonitor

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, curmonitor):
        curlr = self.model.optimizer.param_groups[0]['lr']
        self.curmonitor = curmonitor
        if self.curmonitor is None:
            warnings.warn('errro input of monitor', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(self.curmonitor, self.best):
                self.best = self.curmonitor
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.model.optimizer.param_groups[0]['lr']
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.param_groups[0]['lr'] = new_lr
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
        return curlr

    def in_cooldown(self):
        return self.cooldown_counter > 0

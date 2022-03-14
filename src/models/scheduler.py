from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam
from transformers import ProphetNetForConditionalGeneration

# https://github.com/pytorch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
class InverseSqrtScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup=1000, warmup_init_lr=1e-07, end_lr=1e-4) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.end_lr = end_lr
        self.warmup_init_lr = warmup_init_lr

        self.lr_step = (end_lr - warmup_init_lr) / warmup
        self.decay_factor = end_lr * warmup**0.5

        # Initial learning rate
        self.lr = warmup_init_lr
        self.set_lr()

    def step(self):
        self._step += 1
        self.step_update()
        self.set_lr()
        
    def step_update(self):
        if self._step < self.warmup:
            self.lr = self.warmup_init_lr + self._step * self.lr_step
        else:
            self.lr = self.decay_factor * self._step**-0.5

    def set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = InverseSqrtScheduler(optimizer)
    x = []
    for i in range(5000):
        optimizer.step()
        scheduler.step()
        x.append(optimizer.param_groups[0]['lr'])
    plt.plot(x)
    plt.show()
    
    # iss = InverseSqrtScheduler(Adam)
    # print(iss)
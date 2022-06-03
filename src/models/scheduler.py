from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam
from transformers import ProphetNetForConditionalGeneration

class InverseSqrtScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_updates=1000, warmup_init_lr=1e-07, warmup_end_lr=1e-4) -> None:
        self.optimizer = optimizer

        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr

        self.decay_factor = warmup_end_lr * warmup_updates**0.5
        self.lr = warmup_init_lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        super(InverseSqrtScheduler, self).__init__(optimizer)

    # https://github.com/pytorch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    def get_lr(self):
        num_updates = self._step_count - 1
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates**-0.5
        return [self.lr for group in self.optimizer.param_groups]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
    optimizer = Adam(model.parameters(), weight_decay=0.01)
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
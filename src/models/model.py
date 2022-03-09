import torch
from transformers import ProphetNetForConditionalGeneration
from torch.cuda.amp import autocast

class ProphetNetAutocast(torch.nn.Module):
    def __init__(self, freeze_layers=True) -> None:
        super().__init__()
        self.model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
        self.model = torch.nn.DataParallel(self.model)
        self.model.to('cuda')
        self.model.train()

        # Freeze all but last layer (lm_head: Linear(in_features=1024, out_features=30522, bias=False) )
        if freeze_layers:
            for param in list(self.model.parameters())[:-1]:
                param.requires_grad = False

    @autocast()
    def forward(self, input_ids, attention_mask, labels):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return x.loss.sum()


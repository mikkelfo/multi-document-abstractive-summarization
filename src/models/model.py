from pytorch_lightning import LightningModule
import torch
from transformers import ProphetNetForConditionalGeneration

class ProphetNetModule(LightningModule):
    def __init__(self) -> None:
        super(ProphetNetModule, self).__init__()

        self.prophetnet = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

        # Freeze all but last layer (lm_head: Linear(in_features=1024, out_features=30522, bias=False) )
        # for param in list(self.prophetnet.parameters())[:-1]:
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels):
        x = self.prophetnet(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return x

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.prophetnet.parameters(), lr=0.0001, weight_decay=0.0001)

if __name__ == '__main__':
    pnm = ProphetNetModule()
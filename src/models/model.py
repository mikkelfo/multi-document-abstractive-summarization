import torch
from transformers import ProphetNetForConditionalGeneration, ProphetNetEncoder, ProphetNetDecoder, ProphetNetForCausalLM
from torch.cuda.amp import autocast
import torch.nn.functional as F 

class ProphetNetAutocast(torch.nn.Module):
    def __init__(self, freeze_layers=True) -> None:
        super(ProphetNetAutocast, self).__init__()
        self.model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
        self.model = torch.nn.DataParallel(self.model)
        self.model.to('cuda')

        # Freeze all but last layer (lm_head: Linear(in_features=1024, out_features=30522, bias=False) )
        if freeze_layers:
            for param in list(self.model.parameters())[:-1]:
                param.requires_grad = False

    @autocast()
    def forward(self, input_ids, attention_mask, labels):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return x.loss.sum()


class ProphetNetMulti(torch.nn.Module):
    def __init__(self) -> None:
        super(ProphetNetMulti, self).__init__()
        self.encoder = ProphetNetEncoder.from_pretrained('patrickvonplaten/prophetnet-large-uncased-standalone')
        self.model = ProphetNetForCausalLM.from_pretrained('microsoft/prophetnet-large-uncased')
        assert self.model.config.is_decoder, f"{self.model.__class__} has to be configured as a decoder."

    def get_params(self):
        params = list(self.encoder.parameters()) + list(self.model.parameters())
        return params

    def encoder_pass(self, input_ids, attention_mask):
        last_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        return last_hidden

    def decoder_pass(self, input_ids, attention_mask, encoder_hidden_states):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, use_cache=False).logits

        return logits

    # Taken from https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L2256
    @autocast()
    def forward(self, input_ids, attention_mask, target):
        last_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        logits = self.model(input_ids=target, encoder_hidden_states=last_hidden, use_cache=False).logits
        lprobs = F.log_softmax(logits[0, :, :], dim=-1)

        loss = F.nll_loss(lprobs, target)
        return loss

if __name__ == '__main__':
    model = ProphetNetMulti()
    model.train()
    input_ids = torch.randint(1, 3000, (10, 100))
    attention_mask = torch.ones(10, 100)
    labels = torch.randint(1, 3000, (1, 100))
    model(input_ids, attention_mask, labels)

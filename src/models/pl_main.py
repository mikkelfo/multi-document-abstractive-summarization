from pytorch_lightning import Trainer
from model import ProphetNetModule
from data_module import DataModule

model = ProphetNetModule()
dm = DataModule()
trainer = Trainer(gpus=1, accelerator="gpu", strategy="deepspeed_stage_3_offload", precision=16)
trainer.fit(model, dm)
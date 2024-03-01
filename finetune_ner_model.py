from nemo.collections.nlp.models import TokenClassificationModel

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

config = OmegaConf.load('./configs/token_classification_config.yaml')

# set up dataset
config.model.dataset.data_dir = 'ncbi_data'

# set up GPUs
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 2
config.trainer.accelerator = accelerator
config.trainer.strategy = "ddp"

trainer = pl.Trainer(**config.trainer)

config.model.language_model.pretrained_model_name = "bert-large-uncased"

model_ner = TokenClassificationModel(cfg=config.model, trainer=trainer)

trainer.fit(model_ner)

model_ner.save_to('/workspace/NeMo/demo_ner_model_2.nemo')

print("\n --- Finished training NER model --- \n")
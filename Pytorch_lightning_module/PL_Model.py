from typing import Any

import pytorch_lightning as pl
from analysis.model_zoo.mambaIR import buildMambaIR
from Discriminator import Discriminator


class CycleMambaGAN(pl.LightningModule):
    def __init__(self, train_loader,
                 lr=1e-4,
                 input_shape=(3,64,64)
                 ,val_loader=None):
        super(CycleMambaGAN, self).__init__()
        self.save_hyperparameters()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_generator=buildMambaIR()
        self.hr_generator=buildMambaIR()
        self.lr_discriminator=Discriminator(input_shape=input_shape)
        self.hr_discriminator=Discriminator(input_shape=input_shape)
        self.lr=lr
    def forward(self, lr,hr=None):
        fake_hr=self.hr_generator(lr)
        fake_lr=self.lr_generator(hr)
        if hr is None:
            return fake_hr
        else:
            return fake_lr,fake_hr
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

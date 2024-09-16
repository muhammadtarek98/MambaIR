import pytorch_lightning as pl
import torch.optim.lr_scheduler

from analysis.model_zoo.mambaIR import buildMambaIR
from Discriminator import Discriminator


class CycleMambaGAN(pl.LightningModule):
    def __init__(self, train_loader,
                 learning_rate:float=1e-4,
                 input_shape:tuple=(3,64,64)
                 ,val_loader=None):
        super(CycleMambaGAN, self).__init__()
        self.save_hyperparameters()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_generator=buildMambaIR()
        self.hr_generator=buildMambaIR()
        self.lr_discriminator=Discriminator(input_shape=input_shape)
        self.hr_discriminator=Discriminator(input_shape=input_shape)
        self.learning_rate=learning_rate
        self.automatic_optimization=False

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
    def training_step(self, batch,batch_idx):
        lr_image,hr_image=batch


    def  configure_optimizers(self) :
        self.hparams.lr=self.learning_rate
        hr_generator_optimizer=torch.optim.Adam(params=self.hr_generator.parameters(),
                                                lr=self.learning_rate)
        lr_generator_optimizer=torch.optim.Adam(params=self.lr_generator.parameters(),
                                                lr=self.lr)
        hr_discriminator_optimizer=torch.optim.Adam(params=self.hr_discriminator.parameters(),
                                                    lr=self.lr)
        lr_discriminator_optimizer=torch.optim.Adam(params=self.lr_discriminator.parameters(),
                                                    lr=self.lr)
        hr_generator_schedular=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=hr_generator_optimizer,
                                                                                    T_0=2)
        lr_generator_schedular=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=lr_generator_optimizer,T_0=2)

        hr_discriminator_schedular=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=hr_discriminator_optimizer,T_max=4)

        lr_discriminator_schedular=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=lr_discriminator_optimizer,T_max=4)

        return ([hr_generator_optimizer,lr_generator_optimizer,
                hr_discriminator_optimizer,lr_discriminator_optimizer],
                [hr_generator_schedular,lr_generator_schedular,
                 hr_discriminator_schedular,lr_discriminator_schedular])





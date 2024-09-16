import pytorch_lightning as pl
from analysis.model_zoo.mambaIR import buildMambaIR
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from Discriminator import Discriminator
import torchmetrics
from MambaIR.basicsr.losses.losses import charbonnier_loss


class CycleMambaGAN(pl.LightningModule):
    def __init__(self,batch_size:int, train_loader=None,
                 lr:float=1e-4,
                 input_shape:tuple[int]=(3,64,64)
                 ,val_loader=None,
                 cycle_lambda:int=10,
                 identity_lambda:float=0.5):
        super(CycleMambaGAN, self).__init__()
        self.save_hyperparameters()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cycle_lambda = cycle_lambda
        self.identity_lambda = identity_lambda
        self.lr_generator=buildMambaIR()
        self.hr_generator=buildMambaIR()
        self.lr_discriminator=Discriminator()
        self.hr_discriminator=Discriminator()
        self.learning_rate=lr
        self.l1_loss=torch.nn.L1Loss()
        self.mse_loss=torch.nn.MSELoss()
        self.input_array=torch.randn(batch_size,*input_shape)
        self.logger.log_graph(model=self.lr_generator,input_array=self.input_array)
        self.logger.log_graph(model=self.hr_generator,input_array=self.input_array)
        self.logger.log_graph(model=self.lr_discriminator,input_array=self.input_array)
        self.logger.log_graph(model=self.hr_discriminator,input_array=self.input_array)

    def forward(self, lr:torch.Tensor,hr:torch.Tensor=None):
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

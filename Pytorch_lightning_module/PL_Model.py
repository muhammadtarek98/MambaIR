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

    def discriminator_loss(self,discriminator,generator,fake,real):
        gen=generator(fake)
        real=discriminator(real)
        fake=discriminator(gen.detach())
        real_loss=self.mse_loss(real,torch.ones_like(real))
        fake_loss=self.mse_loss(fake,torch.zeros_like(fake))
        total_loss=(real_loss+fake_loss)/2
        return total_loss

    def generator_loss(self,discriminator,fake):
        disc_fake=discriminator(fake)
        loss=self.mse_loss(disc_fake,torch.ones_like(disc_fake))
        return loss

    def cycle_loss(self,real,cycled):
        loss=self.l1_loss(real,cycled)
        return loss

    def identity_loss(self,real,identity):
        return self.l1_loss(real,identity)

    def training_step(self,batch,batch_idx):
        lr_image,hr_image=batch[0],batch[1]
        fake_hr,fake_lr=self(lr_image,hr_image)
        hr_discriminator_optimizer=self.optimizers()[0]
        lr_discriminator_optimizer=self.optimizers()[1]
        lr_generator_optimizer=self.optimizers()[2]
        hr_generator_optimizer=self.optimizers()[3]

        hr_discriminator_LR_schedular=self.lr_schedulers()[0]
        lr_discriminator_LR_schedular=self.lr_schedulers()[1]
        lr_generator_LR_schedular=self.lr_schedulers()[2]
        hr_generator_LR_schedular=self.lr_schedulers()[3]


        hr_discriminator_optimizer.optimizer.zero_grad()
        lr_discriminator_optimizer.optimizer.zero_grad()

        lr_disc_loss=self.discriminator_loss(discriminator=self.hr_discriminator,
                                             generator=self.hr_generator,
                                             real=hr_image,
                                             fake=hr_image)
        hr_disc_loss=self.discriminator_loss(discriminator=self.lr_discriminator,
                                             fake=hr_image,
                                             real=lr_image,
                                             generator=self.lr_generator)

        total_disc_loss=hr_disc_loss+lr_disc_loss
        self.manual_backward(total_disc_loss,retain_graph=True)
        hr_discriminator_optimizer.step()
        lr_discriminator_optimizer.step()

        hr_discriminator_LR_schedular.step(hr_disc_loss)
        lr_discriminator_LR_schedular.step(lr_disc_loss)

        lr_generator_optimizer.optimizer.zero_grad()
        hr_generator_optimizer.optimizer.zero_grad()
        lr_gen_loss=self.generator_loss(discriminator=self.lr_discriminator,fake=fake_lr)
        hr_gen_loss=self.generator_loss(discriminator=self.hr_discriminator,fake=fake_hr)



        lr_cycled=self.lr_generator(fake_hr)
        hr_cycled=self.hr_generator(fake_lr)
        cycled_lr_loss=self.cycle_loss(real=lr_image,cycled=lr_cycled)
        cycled_hr_loss=self.cycle_loss(real=hr_image,cycled=hr_cycled)
        cycled_loss=(cycled_hr_loss*self.cycle_lambda)+(cycled_lr_loss*self.cycle_lambda)



        lr_identity=self.lr_generator(lr_image)
        hr_identity=self.hr_generator(hr_image)
        lr_identity_loss=self.identity_loss(real=lr_image,identity=lr_identity)
        hr_identity_loss=self.identity_loss(real=hr_image,identity=hr_identity)
        id_loss=(self.identity_lambda*lr_identity_loss)+(self.identity_lambda*hr_identity_loss)


        total_generator_loss=lr_gen_loss+hr_gen_loss+id_loss+cycled_loss
        self.manual_backward(loss=total_generator_loss,retain_graph=True)
        lr_generator_optimizer.step()
        hr_generator_optimizer.step()
        lr_generator_LR_schedular.step(lr_gen_loss)
        hr_generator_LR_schedular.step(hr_gen_loss)
        log={"total_discriminator_loss":total_disc_loss,
             "lr_discriminator_loss":lr_disc_loss,
             "hr_discriminator_loss":hr_disc_loss,
            "total_generator_loss":total_generator_loss,
             "hr_identity_loss":hr_identity_loss,
             "lr_identity_loss":lr_identity_loss,
             "lr_cycle_loss":cycled_lr_loss,
             "hr_cycle_loss":cycled_hr_loss,
        }

        self.log(name="total_discriminator_loss",
                 value=total_disc_loss,on_step=True,
                 on_epoch=True,prog_bar=True
                 )
        self.log(name="lr_discriminator_loss",
                 value=lr_disc_loss,on_step=True,
                 on_epoch=True,prog_bar=True
                 )
        self.log(name="hr_discriminator_loss",
                 value=hr_disc_loss,on_step=True,
                 on_epoch=True,prog_bar=True
                 )
        self.log(name="total_generator_loss",
                 value=total_generator_loss,on_step=True,
                 on_epoch=True,prog_bar=True
                 )

        self.log(name="hr_identity_loss",value=hr_identity_loss,on_step=True,
                 on_epoch=True,prog_bar=True
                 )
        self.log(name="lr_identity_loss",value=lr_identity_loss,on_step=True,
                 on_epoch=True,prog_bar=True
                 )
        self.log(name="lr_cycle_loss",value=cycled_lr_loss,on_step=True,
                 on_epoch=True,prog_bar=True)
        self.log(name="hr_cycle_loss",value=cycled_hr_loss,on_step=True,
                 on_epoch=True,prog_bar=True)

        return logs

    def configure_optimizers(self):
        self.hparams.lr=self.learning_rate
        lr_discriminator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                    params=self.lr_discriminator.parameters())
        hr_discriminator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                    params=self.hr_discriminator.parameters())

        hr_generator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                params=self.hr_generator.parameters())
        lr_generator_optimizer=torch.optim.Adam(lr=self.learning_rate,
                                                params=self.lr_generator.parameters())

        lr_discriminator_LR_schedular=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=lr_discriminator_optimizer)
        hr_discriminator_LR_schedular=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=hr_discriminator_optimizer)

        lr_generator_LR_schedular=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=lr_generator_optimizer,
                                                                             T_max=4)
        hr_generator_LR_schedular=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=hr_generator_optimizer,
                                                                             T_max=4)

        return [[hr_discriminator_optimizer,lr_discriminator_optimizer,
                 lr_generator_optimizer,hr_generator_optimizer],
                [hr_discriminator_LR_schedular,lr_discriminator_LR_schedular,
                 lr_generator_LR_schedular,hr_generator_LR_schedular]]


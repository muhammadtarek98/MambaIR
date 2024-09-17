import pytorch_lightning as pl
import torchinfo
import torch,torchvision
from Dataset import CustomDataset
from Discriminator import Discriminator
import torchmetrics
from MambaIR.analysis.model_zoo.mambaIR import buildMambaIR,buildMambaIR_light
import albumentations as A
class CycleMambaGAN(pl.LightningModule):
    def __init__(self,batch_size:int, train_loader=None,
                 lr:float=1e-4,
                 input_shape:tuple[int]=(3,128,128)
                 ,val_loader=None,
                 cycle_lambda:int=10,
                 identity_lambda:float=0.5):
        super(CycleMambaGAN, self).__init__()
        self.save_hyperparameters()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cycle_lambda = cycle_lambda
        self.identity_lambda = identity_lambda
        self.lr_generator=buildMambaIR_light()
        self.hr_generator=buildMambaIR_light()
        self.lr_discriminator=Discriminator()
        self.hr_discriminator=Discriminator()
        self.learning_rate=lr
        self.l1_loss=torch.nn.L1Loss()
        self.mse_loss=torch.nn.MSELoss()
        self.input_array=torch.randn(batch_size,*input_shape)
        self.automatic_optimization = False


    def forward(self, lr:torch.Tensor,hr:torch.Tensor=None):
        fake_hr=self.hr_generator(lr)
        fake_lr=self.lr_generator(hr)
        if hr is None:
            return fake_hr
        else:
            return fake_hr,fake_lr
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def discriminator_loss(self, discriminator, generator, fake, real):
        gen = generator(fake)
        real = discriminator(real)
        fake = discriminator(gen.detach())
        real_loss = self.mse_loss(real, torch.ones_like(real))
        fake_loss = self.mse_loss(fake, torch.zeros_like(fake))
        total_loss = (real_loss + fake_loss) / 2
        return total_loss

    def generator_loss(self, discriminator, fake):
        disc_fake = discriminator(fake)
        loss = self.mse_loss(disc_fake, torch.ones_like(disc_fake))
        return loss

    def cycle_loss(self, real, cycled):
        loss = self.l1_loss(real, cycled)
        return loss

    def identity_loss(self, real, identity):
        loss = self.l1_loss(real, identity)
        return loss


    def training_step(self, batch,batch_idx):
        lr_image, hr_image = batch[0], batch[1]
        if self.current_epoch==1:
            self.logger.experiment.log_graph(model=self.lr_generator, input_array=self.input_array)
            self.logger.experiment.log_graph(model=self.hr_generator, input_array=self.input_array)
            self.logger.experiment.log_graph(model=self.lr_discriminator, input_array=self.input_array)
            self.logger.experiment.log_graph(model=self.hr_discriminator, input_array=self.input_array)
        fake_hr, fake_lr = self(lr_image, hr_image)
        hr_discriminator_optimizer = self.optimizers()[0]
        lr_discriminator_optimizer = self.optimizers()[1]
        lr_generator_optimizer = self.optimizers()[2]
        hr_generator_optimizer = self.optimizers()[3]

        hr_discriminator_LR_schedular = self.lr_schedulers()[0]
        lr_discriminator_LR_schedular = self.lr_schedulers()[1]
        lr_generator_LR_schedular = self.lr_schedulers()[2]
        hr_generator_LR_schedular = self.lr_schedulers()[3]
        hr_discriminator_optimizer.zero_grad()
        lr_discriminator_optimizer.zero_grad()
        #discriminator loss
        lr_disc_loss = self.discriminator_loss(discriminator=self.hr_discriminator,
                                               generator=self.hr_generator,
                                               fake=lr_image, real=hr_image)
        hr_disc_loss = self.discriminator_loss(discriminator=self.lr_discriminator,
                                               generator=self.lr_generator,
                                               fake=hr_image,
                                               real=lr_image)
        total_disc_loss = hr_disc_loss + lr_disc_loss
        self.manual_backward(total_disc_loss, retain_graph=True)
        hr_discriminator_optimizer.step()
        lr_discriminator_optimizer.step()
        hr_discriminator_LR_schedular.step(lr_disc_loss)
        lr_discriminator_LR_schedular.step(hr_disc_loss)

        #generator loss
        lr_generator_optimizer.zero_grad()
        hr_generator_optimizer.zero_grad()
        lr_gen_loss = self.generator_loss(self.lr_discriminator, fake_lr)
        hr_gen_loss = self.generator_loss(self.hr_discriminator, fake_hr)

        #cycle loss
        cycled_lr = self.lr_generator(fake_hr)
        cycled_hr = self.hr_generator(fake_lr)
        cycled_lr_loss = self.cycle_loss(real=lr_image, cycled=cycled_lr)
        cycled_hr_loss = self.cycle_loss(real=hr_image, cycled=cycled_hr)
        total_cycle_loss=(cycled_hr_loss*self.cycle_lambda)+(cycled_lr_loss*self.cycle_lambda)

        #identity loss
        lr_identity = self.lr_generator(lr_image)
        hr_identity = self.hr_generator(hr_image)
        lr_identity_loss = self.identity_loss(real=lr_image, identity=lr_identity)
        hr_identity_loss = self.identity_loss(real=hr_image, identity=hr_identity)
        total_identity_loss=(self.identity_lambda*lr_identity_loss)+(self.identity_lambda*hr_identity_loss)

        total_gen_loss=lr_gen_loss+hr_gen_loss+total_identity_loss+total_cycle_loss
        self.manual_backward(total_gen_loss,retain_graph=True)
        lr_generator_optimizer.step()
        hr_generator_optimizer.step()

        lr_generator_LR_schedular.step()
        hr_generator_LR_schedular.step()
        self.predictions_logger(hr_cycled=cycled_hr,
                                lr_cycled=cycled_lr,
                                gen_lr_image=fake_hr,
                                gen_hr_image=fake_lr,
                                real_lr_image=lr_image,
                                real_hr_image=hr_image)

        log={"total_discriminator_loss":total_disc_loss,
             "lr_discriminator_loss": lr_disc_loss,
             "total_generator_loss":total_gen_loss,
             "lr_identity_loss":lr_identity_loss,
             "hr_identity_loss":hr_identity_loss,
             "lr_cycle_loss":cycled_lr_loss,
             "hr_cycle_loss":cycled_hr_loss,
             "lr_generator_loss":lr_gen_loss,
             "hr_generator_loss":hr_gen_loss
            }
        for k,v in log.items():
            self.log(name=k,value=v,logger=True,on_epoch=True,prog_bar=True,on_step=True,enable_graph=True)
        return log

    def predictions_logger(self,hr_cycled,lr_cycled,
                           gen_hr_image,real_lr_image,
                           real_hr_image,gen_lr_image)->None:
        real_lr_image=torchvision.utils.make_grid(tensor=real_lr_image,normalize=False)
        hr_cycled=torchvision.utils.make_grid(tensor=hr_cycled,normalize=False)
        lr_cycled=torchvision.utils.make_grid(tensor=lr_cycled,normalize=False)
        real_hr_image=torchvision.utils.make_grid(tensor=real_hr_image,normalize=False)
        gen_hr_image=torchvision.utils.make_grid(tensor=gen_hr_image,normalize=False)
        gen_lr_image=torchvision.utils.make_grid(tensor=gen_lr_image,normalize=False)
        self.logger.experiment.add_image("real low quality image",
                                         real_lr_image,self.current_epoch)
        self.logger.experiment.add_image("cycled high quality image",
                                         hr_cycled,self.current_epoch)
        self.logger.experiment.add_image("cycled low quality image",
                                         lr_cycled,self.current_epoch)
        self.logger.experiment.add_image("real high quality image",
                                         real_hr_image,self.current_epoch)
        self.logger.experiment.add_image("generated low quality image",
                                         gen_lr_image,self.current_epoch)
        self.logger.experiment.add_image("generated high quality image",
                                         gen_hr_image,self.current_epoch)



    def configure_optimizers(self):
        self.hparams.lr = self.learning_rate
        hr_discriminator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                      params=self.hr_discriminator.parameters())
        lr_discriminator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                      params=self.lr_discriminator.parameters())
        lr_generator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                  params=self.lr_generator.parameters())
        hr_generator_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                                  params=self.hr_generator.parameters())
        hr_discriminator_LR_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=hr_discriminator_optimizer)
        hr_generator_LR_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=hr_generator_optimizer,
                                                                               T_max=4)

        lr_discriminator_LR_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=lr_discriminator_optimizer)
        lr_generator_LR_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=lr_generator_optimizer,
                                                                               T_max=4)
        return [[hr_discriminator_optimizer, lr_discriminator_optimizer, lr_generator_optimizer,
                 hr_generator_optimizer],
                [hr_discriminator_LR_schedular, lr_discriminator_LR_schedular,
                 lr_generator_LR_schedular, hr_generator_LR_schedular]]


num_device = 1
root_dir = "/home/muahmmad/projects/Image_enhancement/dataset/underwater_imagenet"
batch_size = 1
device_type = "gpu" if torch.cuda.is_available() else "cpu"  # Use GPU if available

transform = A.Compose(is_check_shapes=False,transforms=[
    A.Resize(height=64, width=64),
    A.ToFloat(),
    A.pytorch.ToTensorV2()],
    additional_targets={'hr_image': 'image'},)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_set = CustomDataset(images_dir=root_dir,
                         device=device,
                         transform=transform)


data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)
logger = pl.loggers.TensorBoardLogger(save_dir="logs",
                                      name="CycleMambaGAN")
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor="total_generator_loss",
    min_delta=0,
    patience=0,
    verbose=False,
    mode="min",
)
checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    save_top_k=1,
    monitor="total_generator_loss",
    mode="min"
)
model=CycleMambaGAN(batch_size=batch_size,train_loader=data_loader)
x=torch.randn(batch_size,3,128,128).to(device,dtype=torch.float32)
model.to(device=device)
trainer = pl.Trainer(
    max_epochs=100,
    val_check_interval=len(data_loader),
    accelerator=device_type,
    devices=num_device,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    enable_progress_bar=True,
    fast_dev_run=False,
)
trainer.fit(model=model,train_dataloaders=data_loader)
#print(model)
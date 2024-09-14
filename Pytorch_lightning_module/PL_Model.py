import pytorch_lightning as pl
from


class CycleMambaGAN(pl.LightningModule):
    def __init__(self, train_loader, val_loader):
        super(CycleMambaGAN, self).__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

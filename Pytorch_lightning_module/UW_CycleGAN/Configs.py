

import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torchvision
out_feature: list = [64, 128, 256, 512]
input_shape: tuple = (3, 720, 720)
batch_size: int = 64
transform = alb.Compose(is_check_shapes=False,transforms=[
    alb.Resize(height=720, width=720),
    alb.ToFloat(),
    #alb.Normalize(mean=[0.24472233,0.50500972,0.4443582],std=[0.20768219,0.24286765,0.2468539 ],always_apply=True),
    alb.pytorch.ToTensorV2()],
    additional_targets={'hr_image': 'image'},)

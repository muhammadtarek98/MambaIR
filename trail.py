from MambaIR.realDenoising.basicsr.models.archs.mambairunet_arch import MambaIRUNet
from MambaIR.basicsr.archs.mambair_arch import MambaIRModel
import torch, cv2, torchvision

model = MambaIRModel(upscale=4,
                     in_chans=3,
                     img_size=64,
                     window_size=16,
                     compress_ratio=3,
                     squeeze_factor=30,
                     conv_scale=0.01,
                     overlap_ratio=0.5,
                     img_range=1.0,
                     depths=[6, 6, 6, 6, 6, 6],
                     embed_dim=180,
                     num_heads=[6, 6, 6, 6, 6, 6],
                     mlp_ratio=2,
                     upsampler="pixelshuffle",
                     resi_connection="1conv")
ckpt = "/home/cplus/projects/m.tarek_master/Image_enhancement/weights/MambaIR/MambaIR-real.pth"
state_dict = torch.load(f=ckpt)
print(state_dict.keys())
model.load_state_dict(state_dict=state_dict["params_ema"])
print(model)

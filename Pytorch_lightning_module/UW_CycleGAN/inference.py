import torch
import cv2
from GANTraining import GAN
import torchvision

def denormalize(tensor, device):
    mean = torch.tensor(data=[0.24472233, 0.50500972, 0.4443582],
                        dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(data=[0.20768219, 0.24286765, 0.2468539],
                       dtype=torch.float32,
                       device=device).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor * 255.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "/home/cplus/projects/m.tarek_master/Image_enhancement/UW_CycleGAN/logs/Underwater_CycleGAN/version_30/checkpoints/epoch=91-step=1184592.ckpt"
model_weights = GAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=device)
model = GAN().to(device)
image = cv2.cvtColor(src=cv2.imread(
    filename="/home/cplus/projects/m.tarek_master/Image_enhancement/Enhancement_Dataset/7393_NF2_f000000.jpg"),
    code=cv2.COLOR_BGR2RGB)
#model.load_state_dict(state_dict=model_weights["state_dict"])
model.eval()
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.24472233, 0.50500972, 0.4443582],
                                     std=[0.20768219, 0.24286765, 0.2468539])
])
input_tensor = transform(image).unsqueeze(0)
input_tensor = input_tensor.to(device)
print(input_tensor.type)
with torch.no_grad():
    prediction = model.generator_lr(input_tensor)
prediction = denormalize(tensor=prediction, device=device)  #.clamp(min=0,max= 1)
prediction = prediction.squeeze()
prediction = prediction.permute(2, 1, 0)
prediction = prediction.detach().cpu().numpy()
prediction = prediction.astype("uint8")
cv2.imshow(mat=prediction, winname="prediction")
cv2.waitKey(0)
cv2.destroyAllWindows()

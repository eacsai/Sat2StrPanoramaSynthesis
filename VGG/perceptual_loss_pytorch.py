import torch
import torch.nn as nn
from torchvision import models
import scipy.io

def compute_error(real, fake):
    return torch.mean(torch.abs(fake - real))

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # 加载 .mat 文件
        self.vgg_mat = scipy.io.loadmat('/public/home/v-wangqw/program/pytorch-CycleGAN-and-pix2pix/models/imagenet-vgg-verydeep-19.mat')
        self.layers = self.vgg_mat['layers'][0]
        features = list(models.vgg19(pretrained=False).features)[:31]
        self.features = nn.ModuleList(features)
        self.load_weights()
    # @profile    
    def load_weights(self):
        # 将权重从.mat转移到 PyTorch 模型
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                weight = torch.tensor(self.layers[i][0][0][2][0][0], dtype=torch.float32).permute(3,2,0,1)
                bias = torch.tensor(self.layers[i][0][0][2][0][1].reshape(-1), dtype=torch.float32)
                
                # Validate dimensions
                assert layer.weight.shape == weight.shape, "Weight shape mismatch"
                assert layer.bias.shape == bias.shape, "Bias shape mismatch"
                
                layer.weight = nn.Parameter(weight)
                layer.bias = nn.Parameter(bias)
    # @profile
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results

# Create instance of the VGG19 model
vgg19 = VGG19().eval()

# @profile
def perceptual_loss(real_img, fake_img):
    imagenet_mean = torch.tensor([123.68, 116.779, 103.939]).view(1,3,1,1).to(real_img.device)
    # Normalize images
    real_img = (real_img + 1.) / 2. * 255. - imagenet_mean
    fake_img = (fake_img + 1.) / 2. * 255. - imagenet_mean
    # Get features
    real_features = [real_img, *vgg19.to(real_img.device)(real_img)]
    fake_features = [fake_img, *vgg19.to(real_img.device)(fake_img)]
    # Calculate losses
    layers_weights = [1.0, 2.6, 4.8, 3.7, 5.6, 1.5]
    total_loss = 0
    for i, weight in enumerate(layers_weights):
        total_loss += compute_error(real_features[i], fake_features[i]) / weight
    
    return total_loss
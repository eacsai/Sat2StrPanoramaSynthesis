import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

def compute_error(real, fake):
    return torch.mean(torch.abs(fake - real))

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = list(vgg19(weights=VGG19_Weights.DEFAULT).to('cuda').features)[:36]
        self.features = nn.ModuleList(features)
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results

def perceptual_loss(real_img, fake_img):
    # Normalize images
    real_img = (real_img + 1.) / 2. * 255.
    fake_img = (fake_img + 1.) / 2. * 255.
    
    # Create instance of the VGG19 model
    vgg19 = VGG19().to(real_img.device)
    vgg19.eval()
    # Get features
    real_features = [real_img, *vgg19(real_img)]
    fake_features = [fake_img, *vgg19(fake_img)]
    
    # Calculate losses
    layers_weights = [1.0, 2.6, 4.8, 3.7, 5.6, 1.5]
    total_loss = 0
    for i, weight in enumerate(layers_weights):
        total_loss += compute_error(real_features[i], fake_features[i]) / weight
    
    return total_loss

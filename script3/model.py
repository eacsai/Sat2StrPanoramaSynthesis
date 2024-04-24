import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .base_model import BaseModel

from geometry.utils import *
from geometry.Geometry_pytorch import *
from VGG.perceptual_loss_pytorch import *
from torch.optim import lr_scheduler

import itertools


EPS = 1e-7

target_height = 128
target_width = 512
aer_size = 256
grd_height = -2
max_height = 6


def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

class PytorchModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        self.opt = opt    
        self.optimizers = []
        self.visual_names = ['real_A', 'fake_B_grd', 'real_B']
        self.loss_names = ['G_GAN', 'G_L1', 'G_perceptual', 'D_real', 'D_fake']
        
        self.generator_outputs_channels = 3
        self.edNet = encoder_decoder(3, generator_outputs_channels=64)
        self.generator = generator(3, self.generator_outputs_channels, self.opt.ngf)
        self.discriminator = discriminator(3)
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        
        if self.opt.heightPlaneNum > 1:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.edNet.parameters(), self.generator.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

    def set_input(self, inputs, targets, polor_inputs):
        self.real_A = inputs.to(self.device).permute(0,3,1,2)
        self.real_B = targets.to(self.device).permute(0,3,1,2)
        self.pol_A = polor_inputs.to(self.device).permute(0,3,1,2)
    
    def create_generator(self, generator_inputs):
        # decoder_1: [batch, 128, 512, ngf * 2] => [batch, 256, 1024, generator_outputs_channels]
        output = self.generator(generator_inputs)
        return output

    def forward(self):
        if self.opt.input_type == 'estimated_height':
            if self.opt.heightPlaneNum > 1:
                self.estimated_height = self.edNet(self.real_A)
            else:
                self.estimated_height = torch.cat([torch.zeros([self.real_A.size(0), 63, self.real_A.size(2), self.real_A.size(3)]),torch.ones([self.real_A.size(0), 1, self.real_A.size(2), self.real_A.size(3)])], axis=1).to('cuda')
            
            argmax_height = torch.argmax(self.estimated_height, dim=1, keepdim=True)
            hight_img = to_pil((argmax_height[0,:,:,:] + 1) / 2)
            hight_img.save(self.opt.name+'_estimated_height.png')
            generator_inputs = geometry_transform(self.real_A, self.estimated_height, target_height, target_width,
                                          self.opt.height_mode, grd_height, max_height, self.opt.method, self.opt.geoout_type, self.opt.dataset)
        
        elif self.opt.input_type == 'pol':
            generator_inputs = self.pol_A
    
        outputs = self.create_generator(generator_inputs)
        
        generator_img = to_pil((generator_inputs[0,:,:,:] + 1) / 2)
        generator_img.save(self.opt.name+'_generator_img.png')
        x_img = to_pil((outputs[0,:,:,:] + 1) / 2)
        x_img.save(self.opt.name+'_x_img.png')
        y_img = to_pil((self.real_B[0,:,:,:] + 1) / 2)
        y_img.save(self.opt.name+'_y_img.png')
        self.fake_B_grd = outputs
    

    def backward_G(self):
        # generator_loss
        # predict_fake => 1
        # abs(targets - outputs) => 0
        predict_fake_grd = self.discriminator(self.fake_B_grd)

        self.loss_G_GAN = self.criterionGAN(predict_fake_grd, True)
        self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B_grd) * self.opt.lambda_L1
        self.loss_G_perceptual = perceptual_loss(self.real_B, self.fake_B_grd) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN * 1 + self.loss_G_perceptual
        
        self.loss_G.backward()

    def backward_D(self):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_real_grd = self.discriminator(self.real_B)
        self.loss_D_real = self.criterionGAN(predict_real_grd, True)
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_fake_grd = self.discriminator(self.fake_B_grd.detach())
        self.loss_D_fake = self.criterionGAN(predict_fake_grd, False)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    
    
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.discriminator, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

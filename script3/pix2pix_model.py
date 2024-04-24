import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torchvision import models
from math import exp
from .vgg import Vgg16
import torch
import torch.nn as nn
from torchvision import models

from geometry.utils import *
from geometry.Geometry_pytorch import *
import itertools


EPS = 1e-7

target_height = 128
target_width = 512
aer_size = 256
grd_height = -2
max_height = 6

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
from .pix2pix_model import gaussian
import torch

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        
        self.vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.edNet = encoder_decoder(3, generator_outputs_channels=64)
        
        self.gan_mode = opt.gan_mode
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D((opt.input_nc), opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimize_ED = torch.optim.Adam(self.edNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.edNet.parameters(), self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def total_variation_loss(self, input):
        # TV loss
        diff_i = torch.sum(torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :]))
        loss = (diff_i + diff_j) / (256 * 256)
        return loss

    def set_input(self, inputs, targets):
        self.real_A = inputs.to(self.device).permute(0,3,1,2)
        self.real_B = targets.to(self.device).permute(0,3,1,2)

    def forward(self):
        self.estimated_height = self.edNet(self.real_A)
        generator_inputs = geometry_transform(self.real_A, self.estimated_height, target_height, target_width,
                                          self.opt.height_mode, grd_height, max_height, self.opt.method, self.opt.geoout_type, self.opt.dataset)
        to_pil = transforms.ToPILImage()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(generator_inputs)  # G(A)
        generator_img = to_pil((generator_inputs[0,:,:,:] + 1) / 2)
        generator_img.save(self.opt.name+'_generator_img.png')
        fake_B_img = to_pil((self.fake_B[0,:,:,:] + 1) / 2)
        fake_B_img.save(self.opt.name+'_fake_B_img.png')

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_A)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        
        self.real_B_features = self.vgg(self.real_B)
        self.fake_B_features = self.vgg(self.fake_B)
        self.loss_G_Perceptual = self.criterionL1(self.fake_B_features[0], self.real_B_features[0]) / 2.6 + self.criterionL1(self.fake_B_features[1], self.real_B_features[1]) / 4.8 + self.criterionL1(self.fake_B_features[2], self.real_B_features[2]) / 3.7 + self.criterionL1(self.fake_B_features[3], self.real_B_features[3]) / 5.6 + self.criterionL1(self.fake_B_features[4], self.real_B_features[4]) * 10 / 1.5
        # self.loss_G_Perceptual = perceptual_loss(self.real_A, self.fake_B)

        # combine loss and calculate gradients
        # if self.gan_mode == 'lpips':
        #     self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_L1 + self.loss_G_Perceptual * self.opt.lambda_vgg
        # else:
        #     self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights


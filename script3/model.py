import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import lpips

from tqdm import tqdm
from .base_model import BaseModel

from geometry.utils import *
from geometry.Geometry_pytorch import *
from VGG.perceptual_loss_pytorch import *
from torch.optim import lr_scheduler

import itertools
import time

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
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.loss_names = ['G_GAN', 'G_Perceptual', 'D_real', 'D_fake']
        self.model_names = ['G', 'D', 'ED']
    
        self.netG_outputs_channels = 3
    
        self.netED = encoder_decoder(3, generator_outputs_channels=64, gpu_ids = self.opt.gpu_ids)
        self.netG = generator(3, self.netG_outputs_channels, self.opt.gpu_ids, self.opt.ngf)
        self.netD = discriminator(6, self.opt.gpu_ids)
    
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        if self.opt.heightPlaneNum > 1:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netED.parameters(), self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

    def set_input(self, inputs, targets, polor_inputs):
        self.real_A = inputs.to(self.device).permute(0,3,1,2)
        self.real_B = targets.to(self.device).permute(0,3,1,2)
        self.pol_A = polor_inputs.to(self.device).permute(0,3,1,2)
    
    def forward(self, isTrain = True):
        if self.opt.input_type == 'estimated_height':
            if self.opt.heightPlaneNum > 1:
                self.estimated_height = self.netED(self.real_A)
            else:
                self.estimated_height = torch.cat([torch.zeros([self.real_A.size(0), 63, self.real_A.size(2), self.real_A.size(3)]),torch.ones([self.real_A.size(0), 1, self.real_A.size(2), self.real_A.size(3)])], axis=1).to('cuda')
            
            if isTrain:
                argmax_height = torch.argmax(self.estimated_height, dim=1, keepdim=True)
                hight_img = to_pil((argmax_height[0,:,:,:] + 1) / 2)
                hight_img.save(self.opt.name+'_estimated_height.png')
                self.generator_inputs = geometry_transform(self.real_A, self.device, self.estimated_height, target_height, target_width, grd_height, max_height, self.opt.method, self.opt.geoout_type, self.opt.dataset)
        
        elif self.opt.input_type == 'pol':
            self.generator_inputs = self.pol_A
    
        self.fake_B = self.netG(self.generator_inputs)
        
        if isTrain:
            generator_img = to_pil((self.generator_inputs[0,:,:,:] + 1) / 2)
            generator_img.save(self.opt.name+'_generator_img.png')
            x_img = to_pil((self.fake_B[0,:,:,:] + 1) / 2)
            x_img.save(self.opt.name+'_x_img.png')
            y_img = to_pil((self.real_B[0,:,:,:] + 1) / 2)
            y_img.save(self.opt.name+'_y_img.png')
    
    def backward_G(self):
        # generator_loss
        # predict_fake => 1
        # abs(targets - outputs) => 0
        fake_AB = torch.cat((self.generator_inputs, self.fake_B), 1)
        predict_fake_grd = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(predict_fake_grd, True)
        # self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B) * self.opt.lambda_L1
        self.loss_G_Perceptual = perceptual_loss(self.real_B, self.fake_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_Perceptual
        
        self.loss_G.backward()
    def backward_D(self):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        real_AB = torch.cat((self.generator_inputs, self.real_B), 1)
        predict_real = self.netD(real_AB.detach())
        self.loss_D_real = self.criterionGAN(predict_real, True)
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        fake_AB = torch.cat((self.generator_inputs, self.fake_B), 1)
        predict_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(predict_fake, False)

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

    def train_step(self, epoch, train_dataloader, visualizer, total_iters):
        self.setup(self.opt)
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        self.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        loop = tqdm(train_dataloader, leave=True)
        loop.set_description(f"Train Epoch {epoch}")
        for step, batch in enumerate(loop):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % self.opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += self.opt.batch_size
            epoch_iter += self.opt.batch_size
            
            aerial = batch.get('aer_image').permute(0,2,3,1).to(self.device)
            polar = batch.get('pol_image').permute(0,2,3,1).to(self.device)
            ground = batch.get('pano_image').permute(0,2,3,1).to(self.device)
            self.set_input(aerial, ground, polar)
            
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

            visualizer.log_image(self.get_current_visuals(), total_iters)  

            if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = self.get_current_losses()
                t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % self.opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                self.save_networks(save_suffix)

            iter_data_time = time.time()
            # sys.exit()
            if epoch % 2 == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.save_networks(epoch)
        
    def test_step(self, epoch, test_dataloader):        
        with torch.no_grad():
            # init Loss
            all_PSNR = []
            all_SSIM = []
            all_Lpips = []
            PSNR = []                       # init PSNR
            SSIM = []                       # init SSIM
            Lpips = []                      # init Lpips
            
            loop = tqdm(test_dataloader, leave=True)
            loop.set_description(f"Test Epoch {epoch}")
            for index, batch in enumerate(loop):
                if index >= 50:  # only apply our model to opt.num_test images.
                    break
                aerial = batch.get('aer_image').permute(0,2,3,1).to(self.device)
                polar = batch.get('pol_image').permute(0,2,3,1).to(self.device)
                ground = batch.get('pano_image').permute(0,2,3,1).to(self.device)               
                self.set_input(aerial, polar, ground)
                self.forward(isTrain=False)
                
                visuals = self.get_current_visuals()  # get image results
                
                loss_PSNR = calculate_psnr(visuals.get('fake_B'), visuals.get('real_B'), 2)
                loss_SSIM = calculate_ssim(visuals.get('fake_B'), visuals.get('real_B'))
                loss_Lpips = self.lpips_fn.forward(visuals.get('fake_B'), visuals.get('real_B')).item()
                
                PSNR.append(loss_PSNR)
                SSIM.append(loss_SSIM)
                Lpips.append(loss_Lpips)
                
                image_path = self.get_image_paths()     # get image paths
                if index % 5 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (index, image_path))
                
                # save to wandb
                metrics = {'PSNR': loss_PSNR, 'SSIM': loss_SSIM, 'Lpips': loss_Lpips}
                message = '(epoch: %d, index: %d)' % (epoch, index)
                for k, v in metrics.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)  # print the message
            # caculate metric
            current_PSNR = sum(PSNR) / len(PSNR)
            current_SSIM = sum(SSIM) / len(SSIM)
            current_Lpips = sum(Lpips) / len(Lpips)
            
            all_PSNR.append(current_PSNR)
            all_SSIM.append(current_SSIM)
            all_Lpips.append(current_Lpips)
            
            best_PSNR = max(all_PSNR)
            best_SSIM = max(all_SSIM)
            best_Lpips = min(all_Lpips)
            
            print(f"current_PSNR: {current_PSNR:.2f}, current_SSIM: {current_SSIM:.2f}, current_Lpips: {current_Lpips:.2f}, best_PSNR: {best_PSNR:.2f}, best_SSIM: {best_SSIM:.2f}, best_Lpips: {best_Lpips:.2f}")
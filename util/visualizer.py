import time
from torchvision.utils import save_image

import os, datetime
from omegaconf import OmegaConf
from .util import save_checkpoint as base_save_checkpoint
from .util import load_checkpoint as base_load_checkpoint

class Visualizer():
    def __init__(self, opt, log_img_freq = 500):
        self.opt = opt
        self.logdir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.log_img_freq = log_img_freq

        self.now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")    
        self.ckptdir = os.path.join(self.logdir, "checkpoints")
        self.cfgdir = os.path.join(self.logdir, "configs")
        self.imgdir = os.path.join(self.logdir, "images")
        self.log_name = os.path.join(self.ckptdir, 'loss_log.txt')
        
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        os.makedirs(self.imgdir, exist_ok=True)
        os.makedirs(os.path.join(self.imgdir, "val"), exist_ok=True)
        with open(self.log_name, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        print("Project config")
        OmegaConf.save(OmegaConf.create(vars(self.opt)),
                        os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
        
    def setup_wandb(self, wandb):
        if not self.log:
            return
        self.wandb = wandb
    
    def save_checkpoint(self, model, loss, opt_disc, opt_ae, epoch):
        if not self.log:
            return
        model_ckpt_name = os.path.join(self.ckptdir, "model-{}.pth".format(epoch))
        base_save_checkpoint(model, opt_ae, model_ckpt_name)
        loss_ckpt_name = os.path.join(self.ckptdir, "loss-{}.pth".format(epoch))
        base_save_checkpoint(loss, opt_disc, loss_ckpt_name)
    
    def load_checkpoint(self, model, loss, opt_disc, opt_ae, model_ckpt, loss_ckpt):
        base_load_checkpoint(model_ckpt, model, opt_ae)
        base_load_checkpoint(loss_ckpt, loss, opt_disc)
    
    def save_image_operation(self, image, filename):
        save_num = min(image.size(0), 4)
        images_to_save = image[:save_num]
        images_to_save = (images_to_save + 1.0) / 2.0
        save_image(images_to_save, filename, nrow=4)
    
    def log_image(self, images, step=0):
        if step % self.opt.display_freq != 0:
            return
        for label, image in images.items():
            self.save_image_operation(image, label + '-' + self.opt.name)
    
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        
        


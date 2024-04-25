from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
import random
import lpips
import os
import ntpath

from script3.pix2pix_model import *
from script3.model import *
from script3.base_options import BaseOptions
from util.visualizer import Visualizer
from util import html

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_psnr(img1, img2, max_value = 255):
    # 计算MSE
    mse = F.mse_loss(img1, img2)
    # 计算MAX_I
    max_value = torch.tensor(max_value).float()
    # 计算PSNR
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret.item()

def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    
    # 设置PyTorch随机种子
    torch.manual_seed(a.seed)

    # 如果你使用CUDA，并且想要从多个进程中获得确定性结果，可以使用以下设置
    # 注意，这可能会降低代码执行效率
    torch.cuda.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)  # 如果使用多个GPU
    # tf.random.set_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    # cmap = np.load('../cmap.npy')
    # print(cmap.shape)
    visualizer = Visualizer(a)   # create a visualizer that display/save images and plots
    # create a website
    web_dir = os.path.join(a.results_dir, a.name, 'test')  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s' % (a.name, 'test'))

    if a.dataset=='CVUSA':
        from load_data.load_data_cvusa import load_examples
    elif a.dataset=='CVACT':
        from load_data.load_data_cvact import load_examples
    elif a.dataset=='CVACThalf':
        from load_data.load_data_cvact_half import load_examples
    elif a.dataset=='CVACTunaligned':
        from load_data.load_data_cvact_unaligned import load_examples

    examples = load_examples(a.batch_size)
    print("examples count = %d" % examples.count)
    train_dataset = examples.train_dataloader
    test_dataset = examples.test_dataloader
    
    model = PytorchModel(a)
    total_iters = 0
    for epoch in range(1, a.n_epochs + a.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        print('============================train==============================')
        model.train()
        model.train_step(epoch, train_dataset, visualizer, total_iters)
        print('============================test==============================')
        with torch.no_grad():
            model.eval()
            # init Loss
            all_PSNR = []
            all_SSIM = []
            all_Lpips = []
            PSNR = []                       # init PSNR
            SSIM = []                       # init SSIM
            Lpips = []                      # init Lpips
            test_step = 0
            
            lpips_fn = lpips.LPIPS(net='alex').cuda()

            for index, data in enumerate(test_dataset):
                if index >= 50:  # only apply our model to opt.num_test images.
                    break
                model.test_step(data)  # unpack data from data loader                
                visuals = model.get_current_visuals()  # get image results
                
                loss_PSNR = calculate_psnr(visuals.get('fake_B_grd'), visuals.get('real_B'), 2)
                loss_SSIM = calculate_ssim(visuals.get('fake_B_grd'), visuals.get('real_B'))
                loss_Lpips = lpips_fn.forward(visuals.get('fake_B_grd'), visuals.get('real_B')).item()
                
                PSNR.append(loss_PSNR)
                SSIM.append(loss_SSIM)
                Lpips.append(loss_Lpips)
                
                image_path = model.get_image_paths()     # get image paths
                if index % 5 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (index, image_path))
                
                # save to wandb
                metrics = {'PSNR': loss_PSNR, 'SSIM': loss_SSIM, 'Lpips': loss_Lpips}
                message = '(epoch: %d, index: %d)' % (epoch, index)
                for k, v in metrics.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)  # print the message
                name = 'test' + str(index)
                webpage.add_header(name)
                ims, txts, links = [], [], []
                webpage.add_images(ims, txts, links, width=a.display_winsize)
                # wandb.log(metrics, step=test_step)
                test_step += 1            
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
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, a.n_epochs + a.n_epochs_decay, time.time() - epoch_start_time))

if __name__ == '__main__':
    parser = BaseOptions().parse()
    a = parser
    EPS = 1e-12
    CROP_SIZE = 256
    main()


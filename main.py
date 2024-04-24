from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
import random

from script3.pix2pix_model import *
from script3.model import *
from script3.base_options import BaseOptions
from util.visualizer import Visualizer

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

    if a.dataset=='CVUSA':
        from load_data.load_data_cvusa import load_examples
    elif a.dataset=='CVACT':
        from load_data.load_data_cvact import load_examples
    elif a.dataset=='CVACThalf':
        from load_data.load_data_cvact_half import load_examples
    elif a.dataset=='CVACTunaligned':
        from load_data.load_data_cvact_unaligned import load_examples

    examples = load_examples(a.mode, a.batch_size)
    print("examples count = %d" % examples.count)
    dataset = examples.dataloader
    
    total_iters = 0                # the total number of training iterations
    train_step = 0
    if a.model_type == 'primary':
        model = PytorchModel(a)
    elif a.model_type == 'pix2pix':
        model = Pix2PixModel(a)

    for epoch in range(1, a.n_epochs + a.n_epochs_decay + 1):
        print('============================train==============================')
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % a.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += a.batch_size
            epoch_iter += a.batch_size
            
            inputs = data.get('aer_image').permute(0,2,3,1)
            pol_inputs = data.get('pol_image').permute(0,2,3,1)
            targets = data.get('pano_image').permute(0,2,3,1)
            model.set_input(inputs, targets, pol_inputs)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % a.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % a.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % a.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / a.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if a.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)
                    train_step += 1

            # if total_iters % a.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if a.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % 2 == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, a.n_epochs + a.n_epochs_decay, time.time() - epoch_start_time))

if __name__ == '__main__':
    parser = BaseOptions().parse()
    a = parser
    EPS = 1e-12
    CROP_SIZE = 256
    main()


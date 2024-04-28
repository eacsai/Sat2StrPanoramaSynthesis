from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
import random

from script3.model import *
from script3.base_options import BaseOptions
from util.visualizer import Visualizer
from load_data.load_data_cvusa import load_examples

def main():
    if opt.seed is None:
        opt.seed = random.randint(0, 2**31 - 1)
    
    # 设置PyTorch随机种子
    torch.manual_seed(opt.seed)
    # 如果你使用CUDA，并且想要从多个进程中获得确定性结果，可以使用以下设置
    # 注意，这可能会降低代码执行效率
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)  # 如果使用多个GPU
    # tf.random.set_seed(a.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    # cmap = np.load('../cmap.npy')
    # print(cmap.shape)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    examples = load_examples(opt.batch_size)
    print("examples count = %d" % examples.count)
    train_dataset = examples.train_dataloader
    test_dataset = examples.test_dataloader
    
    model = PytorchModel(opt)
    total_iters = 0
    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        print('============================train==============================')
        model.train()
        model.train_step(epoch, train_dataset, visualizer, total_iters)
        print('============================test==============================')
        model.eval()
        model.test_step(epoch, test_dataset)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

if __name__ == '__main__':
    opt = BaseOptions().parse()
    main()


import collections
import os.path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random
import math

# 设置随机种子
seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

Examples = collections.namedtuple("Examples", "paths, dataloader, count, steps_per_epoch")

class ImageDataset(Dataset):
    def __init__(self, aer_list, pano_list, tanpolar_list, polar_list):
        self.aer_list = aer_list
        self.pano_list = pano_list
        self.tanpolar_list = tanpolar_list
        self.polar_list = polar_list
        self.preprocess = transforms.Compose([
            # 此步骤后，像素值会在[0, 1]范围内
            transforms.ToTensor(),
            # Normalize步骤会将[0, 1]范围的值转换为[-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # 在这里添加其他必要的转换，例如缩放图像等
            transforms.Resize((128, 512), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.aer_list)
    
    def __getitem__(self, idx):
        aer_image = Image.open(self.aer_list[idx]).convert('RGB')
        pano_image = Image.open(self.pano_list[idx]).convert('RGB')
        tanpolar_image = Image.open(self.tanpolar_list[idx]).convert('RGBA')
        polar_image = Image.open(self.polar_list[idx]).convert('RGBA')

        # 应用预处理
        aer_image = self.preprocess(aer_image)
        pano_image = self.preprocess(pano_image)
        tanpolar_image = self.preprocess(tanpolar_image)
        polar_image = self.preprocess(polar_image)

        return aer_image, pano_image, tanpolar_image, polar_image


# pytorch version of CVUSA loader
def load_examples(mode='train', batch_size=2):
    
    img_root = '/public/home/v-wangqw/program/CVACT/'
    if mode=='train':
        file_list = os.path.join(img_root, 'splits/train-19zl.csv')
    elif mode=='test':
        file_list = os.path.join(img_root, 'splits/val-19zl.csv')
    
    data_list = []
    with open(file_list, 'r') as f:
        for line in f:
            data = line.split(',')
            # data_list.append([img_root + data[0], img_root + data[1], img_root + data[2][:-1]])
            data_list.append([img_root + data[0], img_root + data[1],
                            # img_root + data[2][:-1].replace('annotations', 'annotations_visualize'),
                            img_root + data[0].replace('bingmap/19', 'a2g').replace('jpg', 'png'),
                            img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')])

    aer_list = [item[0] for item in data_list]
    pano_list = [item[1] for item in data_list]
    # mask_list = [item[2] for item in data_list]
    tanpolar_list = [item[2] for item in data_list]
    polar_list = [item[3] for item in data_list]
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(aer_list, pano_list, tanpolar_list, polar_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    steps_per_epoch = int(math.ceil(len(data_list) / batch_size))
        
    return Examples(
        paths=img_root + data[0],
        dataloader=dataloader,
        count=len(data_list),
        steps_per_epoch=steps_per_epoch,
    )

if __name__ == '__main__':
    examples = load_examples(mode='train', batch_size=2)
    for i, data in enumerate(examples.dataloader):
        aer_image, pano_image, tanpolar_image, polar_image = data
        print(aer_image.shape, pano_image.shape, tanpolar_image.shape, polar_image.shape)
        if i == 0:
            break
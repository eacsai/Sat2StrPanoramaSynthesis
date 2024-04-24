import collections
import os.path
import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import math
import scipy.io as sio

Examples = collections.namedtuple("Examples", "paths, dataloader, count, steps_per_epoch")

class ImageDataset(Dataset):
    def __init__(self, aer_list, pano_list):
        self.aer_list = aer_list
        self.pano_list = pano_list
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

        # 应用预处理
        aer_image = self.preprocess(aer_image)
        pano_image = self.preprocess(pano_image)
        pano_path = self.pano_list[idx]

        return pano_path, aer_image, pano_image


# pytorch version of CVUSA loader
def load_examples(mode='train', batch_size=2):
    
    img_root = '/public/home/v-wangqw/program/CVACT/'
    allDataList = os.path.join(img_root, 'ACT_data.mat')

    exist_aer_list = os.listdir(img_root + 'satview_polish')
    exist_grd_list = os.listdir(img_root + 'streetview')
    exist_polar_list = os.listdir(img_root + 'streetview')
    
    __cur_allid = 0  # for training
    
    # load the mat
    anuData = sio.loadmat(allDataList)
    
    data_list = []
    for i in range(0, len(anuData['panoIds'])):
        # grd_id_align = img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'
        # sat_id_ori = img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.png'
        grd_id_align = anuData['panoIds'][i] + '_grdView.jpg'
        sat_id_ori = anuData['panoIds'][i] + '_satView_polish.jpg'
        data_list.append([grd_id_align, sat_id_ori])
    
    if mode=='train':
        training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        trainNum = len(training_inds)
        trainList = []
        for k in range(trainNum):
            trainList.append(data_list[training_inds[k][0]])
        pano_list = [img_root + 'streetview/' + item[0] for item in trainList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        aer_list = [img_root + 'satview_polish/' + item[1] for item in trainList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        polar_list = [img_root + 'polar_polish/' + item[1] for item in trainList if item[0] in exist_grd_list and item[1] in exist_aer_list]

    
    else:
        val_inds = anuData['valSet']['valInd'][0][0] - 1
        valNum = len(val_inds)
        valList = []
        for k in range(valNum):
            valList.append(data_list[val_inds[k][0]])
        pano_list = [img_root + 'streetview/' + item[0] for item in valList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        aer_list = [img_root + 'satview_polish/' + item[1] for item in valList if item[0] in exist_grd_list and item[1] in exist_aer_list]
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一个在 CUDA 设备上的生成器
    generator = torch.Generator(device=device)
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(aer_list, pano_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    steps_per_epoch = int(math.ceil(len(data_list) / batch_size))
        
    return Examples(
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
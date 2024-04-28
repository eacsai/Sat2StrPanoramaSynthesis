import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

to_pil = transforms.ToPILImage()

def normlize(X):

    # 归一化到 [0, 1]
    X_normalized = X / 256

    # 转换到 [-1, 1]
    X_scaled = 2 * X_normalized - 1
    return X_scaled

def normalize_2d_array(arr):
    # 计算数组的最小值和最大值
    min_val = np.min(arr)
    max_val = np.max(arr)

    # 使用线性缩放公式归一化到 [-1, 1]
    normalized_arr = 2 * ((arr - min_val) / (max_val - min_val)) - 1

    return normalized_arr

def geometry_transform(aer_imgs, device, estimated_height, target_height, target_width, grd_height, max_height,
                       method='column', geoout_type='image', dataset='CVUSA'):
    '''
    :param aer_imgs:
    :param estimated_height:
    :param mode: if estimated_height.channel ==1, type belongs to {'hole', 'column'};
           otherwise if estimated_height.channel>1, type belongs to {'radiusPlaneMethod', 'heightPlaneMethod'}
    The following two parameters are only needed if mode is 'radiusPlaneMethod'.
    :param method: select from {'column', 'point'}.
                    'column' means: for each point in overhead view, we poject it and the points under it to the grd view
                                    we use cusum to mimic this process
                    'point' means we only project the points in the overhead view image to the grd view.
    :param geoout_type: select from {'volume', 'image'}.
    :return:
    '''
    # PlaneNum = estimated_height.get_shape().as_list()[-1]
    # if height_channel==1:
    # if mode == 'heightPlaneMethod':
    output = MultiPlaneImagesAer2Grd_height(aer_imgs, device, estimated_height, target_height, target_width, grd_height,
                                                max_height, method, geoout_type, dataset)
    # elif mode == 'radiusPlaneMethod':
    #     output = MultiPlaneImagesAer2Grd_height(aer_imgs, estimated_height, target_height, target_width,
    #                                             grd_height, max_height, method, geoout_type, dataset)
    return output

def MultiPlaneImagesAer2Grd_height(signal, device, estimated_height, target_height, target_width, grd_height=-2, max_height=30,
                                   method='column', geoout_type='image', dataset='CVUSA'):
    PlaneNum = estimated_height.shape[1]

    # 使用 torch.full 创建张量
    if method == 'column':
        estimated_height = torch.cumsum(estimated_height, axis=1)
        # the maximum plane corresponds to grd plane
    batch, C, H, W = signal.shape
    assert (H == W)
    
    i = torch.arange(0, (target_height*2), device=device)
    j = torch.arange(0, target_width, device=device)
    jj, ii = torch.meshgrid(j, i, indexing='xy')

    if dataset == 'CVUSA':
        f = H/55
    elif dataset == 'CVACT' or dataset == 'CVACThalf':
        f = H/(50*206/256)
    elif dataset == 'CVACTunaligned':
        f = H/50
    elif dataset == 'OP':
        f = H/100

    # f = H/144

    tanii = torch.tan(ii * np.pi / (target_height*2))
    sin_jj = torch.sin(jj * 2 * np.pi / target_width)
    cos_jj = torch.cos(jj * 2 * np.pi / target_width)
    
    images_list = []
    alphas_list = []
    m = target_height 
    n = int(target_height/2)

    # images_list_volume = []
    for i in range(PlaneNum):
        warp = torch.full((batch, (target_height*2), target_width, 2), -10, dtype=torch.float32, device=device)
        z = grd_height + (max_height-grd_height) * i/PlaneNum
        v = H / 2. - f * z * tanii * sin_jj
        u = H / 2. + f * z * tanii * cos_jj
        v = normlize(v)
        u = normlize(u)

        if z > 0:
            warp[:, -m:, :, 0] = u[-m:, :]
            warp[:, -m:, :, 1] = v[-m:, :]
        else:
            warp[:, :m, :, 0] = u[0:m, :]
            warp[:, :m, :, 1] = v[0:m:, :]
        warp = warp[:,n:-n, ...]
        # images_prob = tf.contrib.resampler.resampler(signal*estimated_height[..., i:i+1], warp)
        # images = tf.contrib.resampler.resampler(signal, warp)
        # alphas = tf.contrib.resampler.resampler(estimated_height[..., i:i + 1], warp)
        # 重采样操作
        images = F.grid_sample(signal, warp, mode='bilinear', padding_mode='zeros', align_corners=False)
        alphas = F.grid_sample(estimated_height[:, i:i + 1, ...], warp, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        images_list.append(images)
        alphas_list.append(alphas)

        # images_list_volume.append(images_prob)

    if geoout_type == 'volume':
        return torch.cat([images_list[i]*alphas_list[i] for i in range(PlaneNum)], axis=-1)
        # return tf.concat(images_list, axis=-1) * tf.concat(alphas_list, axis=-1)  # shape = [batch, target_height, target_width, channel*PlaneNum]
    elif geoout_type == 'image':
        for i in range(PlaneNum):
            rgb = images_list[i]
            a = alphas_list[i]
            if i == 0:
                output = rgb * a
            else:
                rgb_by_alpha = rgb * a
                output = rgb_by_alpha + output * (1 - a)

        return output  # shape = [batch, target_height, target_width, channel]

    # batch_image = tf.stack(images_list, axis=-1)
    #
    # batch_mulplanes = tf.reshape(batch_image, [-1, target_height, target_width, C*PlaneNum])
    #
    # return batch_mulplanes

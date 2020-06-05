import numpy as np
import cv2
import torch
from utilss import crop_image, normalize_, color_jittering_, lighting_
from utils import random_crop, draw_gaussian, gaussian_radius
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
from Net import HourglassNet
from torch import optim
from torch import nn
from PIL import Image

def _full_image_crop(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:,1] += border[0]
    detections[:,0] += border[2]
    return image, detections


def _resize_image(image, detections, size):
    detections = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections


def _clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0] = np.clip(detections[:, 0], 0, width - 1)
    detections[:, 1] = np.clip(detections[:, 1], 0, height - 1)
    keep_inds = ((detections[:, 2] - detections[:, 0]) > 0) & \
                ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

class MydataSet(Dataset):
    def __init__(self):
        super(MydataSet, self).__init__()
        with open(r'D:\MASKpicture\datas\train.txt') as f:
            datas = f.readlines()
            self.datasets = [data.split() for data in datas]
            self.datasets = self.datasets
            self.mean = np.array([0.40789655, 0.44719303, 0.47026116])
            self.std = np.array([0.2886383, 0.2740816, 0.27809834])
            self.categories = 6
            self.input_size = [512, 512]
            self.output_size = [128, 128]
            self.gaussian_bump = True
            self.gaussian_iou = 0.7
            self.gaussian_rad = -1

    def __getitem__(self,index):
        try:
            data = self.datasets[index]
            image_path = os.path.join(r'D:\MASKpicture\datas\zitai_datasets', data[0])
            points = np.split(np.array([int(x) for x in data[1:]]),len(data[1:]) // 2)
            detections = np.array(points,dtype=np.float64)
            heatmaps = np.zeros((self.categories, self.output_size[0], self.output_size[1]), dtype=np.float32)
            image = cv2.imread(image_path)
            # reading detections
            # cropping an image randomly
            image, detections = _full_image_crop(image, detections)  # 填充成一个方形
            image, detections = _resize_image(image, detections, self.input_size)  # resize为512
            # detections = _clip_detections(image, detections)
            width_ratio = self.output_size[1] / self.input_size[1]
            height_ratio = self.output_size[0] / self.input_size[0]
            image = image.astype(np.float32) / 255.
            normalize_(image, self.mean, self.std)
            image = image.transpose((2, 0, 1))
            for ind, detection in enumerate(detections):
                x, y = detection[0], detection[1]
                # 计算图片在经过下采样以后的实际的点坐标
                fx = (x * width_ratio)
                fy = (y * height_ratio)
                x = int(fx)
                y = int(fy)
                # 生成高斯heatmap
                if self.gaussian_bump:
                    width = 10
                    height = 10
                    if self.gaussian_rad == -1:
                        # 计算高斯半径
                        radius = gaussian_radius((height, width), self.gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = self.gaussian_rad
                    draw_gaussian(heatmaps[ind], [x, y], radius)
            image = torch.from_numpy(image)
            heatmaps = torch.from_numpy(heatmaps)
            return image, heatmaps
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.datasets)

# def focalloss(y_hat,y):
#     ones = torch.ones_like(y,dtype=y.dtype)
#     loss = torch.where(y == ones,((1.0 - y_hat)**2) * torch.log(y_hat),((1 - y)**4) * (y_hat**2) * torch.log(1.0 - y_hat)).sum() / 6
#     return torch.abs(loss)

def focalloss(y_hat,y):
    zeros = torch.zeros_like(y)
    pos_corr = torch.where(y > zeros,y - y_hat,zeros)
    neg_corr = torch.where(y > zeros,zeros,y_hat)
    loss = -0.25 * (pos_corr ** 2) * torch.log(y_hat) - \
           (1 - 0.25) * (neg_corr ** 2) * torch.log(1.0 - y_hat)
    return torch.sum(loss)
from Focalloss import FocalLoss
if __name__ == '__main__':
    model = HourglassNet(nStacks=6, nModules=4, nFeat=128, nClasses=6)#torch.load('horglass_purple_ear.pth')#
    model.cuda()
    model.train()
    datasets = MydataSet()
    dataloader = DataLoader(datasets,batch_size=1,shuffle=True,num_workers=4)
    optimizer = optim.Adam(model.parameters())
    criticizer = nn.BCELoss()
    Loss1 = 100
    Loss = 0
    count = 0
    for epoch in range(10000000000):
        num = 0
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)[-1]
            loss = criticizer(outputs,y)#focalloss(outputs,y)#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('ecpoch {}, num: {}:'.format(epoch, num), loss)
            Loss += loss
            num += 1
            count += 1
        Loss = Loss / count
        count = 0
        # if (epoch + 1) % 1 == 0:
        if Loss < Loss1:
            Loss1 = Loss

            torch.save(model, 'length_point_al_{}_{}'.format(epoch, Loss1))#horglass_purple_ear1.pth
            Loss = 0
            print('save model；length_point_al_{}_{}'.format(epoch, Loss1))
        else:
            Loss = 0

# if __name__ == '__main__':
#     model = torch.load('horglass_red_h.pth')#HourglassNet(nStacks=2, nModules=4, nFeat=128, nClasses=6)
#     model.cuda()
#     model.eval()
#     datasets = MydataSet()
#     dataloader = DataLoader(datasets,batch_size=1,shuffle=False,num_workers=4)
#     for x, y in dataloader:
#         y = y[0]
#         x = x.cuda()
#         outputs = model(x)[-1][0]
#         i = 0
#         loss = 0
#         for output in outputs:
#             o = y[i]
#             output = output.cpu().detach().numpy()
#             o = o.detach().numpy()
#             index1 = np.where(output >= np.max(output))
#             index2 = np.where(o >= np.max(o))
#             x_pre,y_pre = index1[0],index1[1]
#             x_real, y_real = index2[0], index2[1]
#             l = np.sqrt((x_pre / 128 - x_real / 128)**2 + (y_pre / 128 - y_real / 128)**2)
#             loss += l
#             i += 1
#         loss /= 6
#         print(1 - loss)

# if __name__ == '__main__':
#     model = torch.load('horglass_purple_ear.pth',map_location='cpu')#HourglassNet(nStacks=2, nModules=4, nFeat=128, nClasses=6)
#     # model.cuda()
#     model.eval()
#     for dir in os.listdir(r'D:\ear\data'):
#         # try:
#             image_path = r'D:\ear\data\{}'.format(dir)
#             # image_path = r'D:\traindatasets\train_balck\{}'.format(dir)
#             detections = np.ones(shape=(2,2))
#             image = cv2.imread(image_path)
#             # reading detections
#             # cropping an image randomly
#             image, detections = _full_image_crop(image, detections)  # 填充成一个方形
#             image, detections = _resize_image(image, detections,[512,512])  # resize为512
#             x = image.astype(np.float32) / 255.
#             mean = np.array([0.40789655, 0.44719303, 0.47026116])
#             std = np.array([0.2886383, 0.2740816, 0.27809834])
#             normalize_(x, mean, std)
#             x = np.expand_dims(x.transpose((2, 0, 1)),axis=0)
#             x = torch.from_numpy(x)
#             # x = x.cuda()
#             outputs = model(x)[-1][0]
#             for output in outputs:
#                 output = output.cpu().detach().numpy()
#                 index1 = np.where(output >= np.max(output))
#                 y_pre, x_pre = index1[0], index1[1]
#                 x_pre *= 4
#                 y_pre *= 4
#                 image = cv2.circle(image, (x_pre, y_pre), 3, (0, 255, 0), -1)
#             # cv2.imwrite(r'D:\360Downloads\12_pre\{}'.format(dir),image)
#             # print(dir)
#                 cv2.imshow('',image)
#                 cv2.waitKey(0)
#             # exit()
#         # except:
#         #     print(dir,'*'*20)
#             pass
# from pigweights import deeplabv3_resnet50
# from skimage.transform import resize
# from pigweights import get_h
# import torch.nn.functional as F
# from math import *
# if __name__ == '__main__':
#     model = torch.load('horglass_purple_ear.pth',map_location='cuda')
#     model.cuda()
#     model.eval()
#     net_paper = deeplabv3_resnet50(num_classes=1)
#     net_paper.cuda()
#     net_paper.load_state_dict(torch.load(r'D:\UnetParams_paper\paramdeeplab91.pth', map_location='cuda'))
#     net_paper.eval()
#     for dir in os.listdir(r'pic'):
#         # try:
#             image_path = 'pic\{}'.format(dir)
#             # image_path = r'D:\traindatasets\train_balck\{}'.format(dir)
#             detections = np.ones(shape=(2,2))
#             image = cv2.imread(image_path)
#
#             img_h, img_w, _ = image.shape
#             dim_diff = np.abs(img_h - img_w)
#             pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
#             pad = ((pad1, pad2), (0, 0), (0, 0)) if img_h <= img_w else ((0, 0), (pad1, pad2), (0, 0))
#             input_img = np.pad(image, pad, 'constant', constant_values=128) / 255.
#             padded_h, padded_w, _ = input_img.shape
#
#             input_img1 = resize(input_img, (490, 490, 3), mode='reflect')
#             input_data1 = np.expand_dims(np.transpose(input_img1, (2, 0, 1)), axis=0)
#             imagedata1 = torch.Tensor(np.array(input_data1, dtype='float32'))
#             imagedata1 = imagedata1.cuda()
#
#             paper = F.sigmoid(net_paper(imagedata1)['out'])
#             paper = paper.cpu().detach().numpy()
#             paper = np.reshape(paper, (490, 490))
#
#             mask_paper = paper > 0.8
#             mask_nopaper = paper < 0.8
#             indexs_paper = np.array(np.where(mask_paper == True))
#             indexs_paper = np.stack(indexs_paper, axis=1)
#             paper[indexs_paper] = 255
#             paper[mask_nopaper] = 0
#             cv2.imwrite('test.jpg', paper)
#             h = get_h('test.jpg') * 512 / 490
#
#             # reading detections
#             # cropping an image randomly
#             image, detections = _full_image_crop(image, detections)  # 填充成一个方形
#             image, detections = _resize_image(image, detections,[512,512])  # resize为512
#             x = image.astype(np.float32) / 255.
#             mean = np.array([0.40789655, 0.44719303, 0.47026116])
#             std = np.array([0.2886383, 0.2740816, 0.27809834])
#             normalize_(x, mean, std)
#             x = np.expand_dims(x.transpose((2, 0, 1)),axis=0)
#             x = torch.from_numpy(x)
#             x = x.cuda()
#             outputs = model(x)[-1][0]
#             points = []
#             for output in outputs:
#                 output = output.cpu().detach().numpy()
#                 index1 = np.where(output >= np.max(output))
#                 y_pre, x_pre = index1[0], index1[1]
#                 x_pre *= 4
#                 y_pre *= 4
#                 image = cv2.circle(image, (x_pre, y_pre), 3, (0, 255, 0), -1)
#                 cv2.imwrite(r'D:\360Downloads\12_pre\{}'.format(dir),image)
#                 points.append(x_pre)
#                 points.append(y_pre)
#             cv2.imshow('', image)
#             cv2.waitKey(0)
#             disdence = sqrt((points[0] - points[2])**2 + (points[1] - points[3])**2)
#             print(297 / h * disdence / 10)
#             pass
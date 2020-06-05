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
        with open('pig_count.txt') as f:
            datas = f.readlines()
            self.datasets = [data.split() for data in datas]
            self.datasets = self.datasets
            self.mean = np.array([0.40789655, 0.44719303, 0.47026116])
            self.std = np.array([0.2886383, 0.2740816, 0.27809834])
            self.input_size = [512, 512]
            self.output_size = [128, 128]
            self.gaussian_bump = True
            self.gaussian_iou = 0.7
            self.gaussian_rad = -1

    def __getitem__(self,index):
            data = self.datasets[index]
            image_path = os.path.join(r'C:\Users\Administrator\Desktop\pig\JPEGImages', data[0])
            points = np.split(np.array([int(x) for x in data[1:]]),len(data[1:]) // 2)
            detections = np.array(points,dtype=np.float64)
            heatmaps = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
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
                    draw_gaussian(heatmaps, [x, y], radius)
            heatmaps = np.expand_dims(heatmaps,axis=0)
            image = torch.from_numpy(image)
            heatmaps = torch.from_numpy(heatmaps)
            return image, heatmaps

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

# if __name__ == '__main__':
#     model = torch.load('count_pigs.pth')#HourglassNet(nStacks=8, nModules=4, nFeat=128, nClasses=1)#
#     model.cuda()
#     model.train()
#     datasets = MydataSet()
#     dataloader = DataLoader(datasets,batch_size=2,shuffle=True,num_workers=4)
#     optimizer = optim.Adam(model.parameters())
#     criticizer = nn.BCELoss()
#     for epoch in range(10000000000):
#         for x, y in dataloader:
#             x = x.cuda()
#             y = y.cuda()
#             outputs = model(x)[-1]
#             loss = criticizer(outputs,y)#focalloss(outputs,y)#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print(loss)
#         torch.save(model,'count_pigs.pth')
#         print('save')

# if __name__ == '__main__':
#     model = torch.load('count_pigs.pth')#HourglassNet(nStacks=2, nModules=4, nFeat=128, nClasses=6)
#     model.cuda()
#     model.eval()
#     for dir in os.listdir(r'C:\Users\Administrator\Desktop\pig\JPEGImages'):
#             image_path = r'C:\Users\Administrator\Desktop\pig\JPEGImages\{}'.format(dir)
#             # image_path = r'D:\traindatasets\train_balck\{}'.format(dir)
#             detections = np.ones(shape=(6,2))
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
#             x = x.cuda()
#             output = model(x)[-1][0][0]
#             output = output.cpu().detach().numpy() * 255
#             index1 = np.where(output >= (np.max(output) / 2))
#             index2 = np.where(output < (np.max(output) / 2))
#             output[index1] = 255
#             output[index2] = 0
#             kernel = np.ones((1, 1), np.uint8)
#             output = cv2.erode(output, kernel)  # 腐蚀
#             # y_pre, x_pre = index1[0], index1[1]
#             # x_pre *= 4
#             # y_pre *= 4
#             # image = cv2.circle(image, (x_pre, y_pre), 3, (0, 255, 0), -1)
#             # # # cv2.imwrite(r'D:\__MACOSX\predict_20\{}'.format(dir),image)
#             # # # print(dir)
#             cv2.imshow('',output)
#             cv2.waitKey(0)
#             # # exit()


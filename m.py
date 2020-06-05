import numpy as np
import cv2
import torch
from utilss import crop_image, normalize_
import os
from math import *

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

import csv
if __name__ == '__main__':
    model = torch.load('horglass.pth')#HourglassNet(nStacks=2, nModules=4, nFeat=128, nClasses=6)
    model.cuda()
    model.eval()
    outputs = open(r'体尺18.csv', 'w', newline='')
    csv_write = csv.writer(outputs, dialect='excel')
    csv_write.writerow(['图片名', '体长','胸围'])
    for dir in os.listdir(r'D:\traindatasets\18'):
        try:
            image_path = r'D:\traindatasets\18\{}'.format(dir)
            detections = np.ones(shape=(6,2))
            image = cv2.imread(image_path)
            # reading detections
            # cropping an image randomly
            image, detections = _full_image_crop(image, detections)  # 填充成一个方形
            image, detections = _resize_image(image, detections,[512,512])  # resize为512
            x = image.astype(np.float32) / 255.
            mean = np.array([0.40789655, 0.44719303, 0.47026116])
            std = np.array([0.2886383, 0.2740816, 0.27809834])
            normalize_(x, mean, std)
            x = np.expand_dims(x.transpose((2, 0, 1)),axis=0)
            x = torch.from_numpy(x)
            x = x.cuda()
            outputs = model(x)[-1][0]
            points = []
            for output in outputs:
                output = output.cpu().detach().numpy()
                index1 = np.where(output >= np.max(output))
                y_pre, x_pre = index1[0], index1[1]
                points.append([x_pre,y_pre])
            index = points
            # 前肘
            elbow = [index[0][0],index[0][1]]
            #屁股
            hip = [index[1][0],index[1][1]]
            #上背
            back = [index[2][0],index[2][1]]
            #腰
            waist = [index[3][0],index[3][1]]
            #纸
            lefttop = [index[4][0],index[4][1]]
            rightbottow = [index[5][0],index[5][1]]
            rate = 36.37 / sqrt((lefttop[0] - rightbottow[0]) ** 2 + (lefttop[1] - rightbottow[1]) ** 2)
            bodylen_real = sqrt((elbow[0] - hip[0]) ** 2 + (elbow[1] - hip[1]) ** 2) * rate
            chest_real = sqrt((back[0] - waist[0]) ** 2 + (back[1] - waist[1]) ** 2) * rate
            # real = bodylen_real * chest_real
            csv_write.writerow([dir, str(bodylen_real),str(chest_real)])
            print(dir)
        except:
            pass
        # if bodylen_real * chest_real < 700:
        #     real_weight = chest_real * 2 * bodylen_real / 200
        # elif bodylen_real * chest_real >= 700 and bodylen_real * chest_real < 1000:
        #     real_weight = chest_real * 2 * bodylen_real / 190
        # elif bodylen_real * chest_real >= 1000 and bodylen_real * chest_real < 1200:
        #     real_weight = chest_real * 2 * bodylen_real / 160
        # elif bodylen_real * chest_real >= 1200 and bodylen_real * chest_real < 1400:
        #     real_weight = chest_real * 2 * bodylen_real / 156
        # elif bodylen_real * chest_real >= 1400 and bodylen_real * chest_real < 1600:
        #     real_weight = chest_real * 2 * bodylen_real / 142
        # elif bodylen_real * chest_real >= 1400 and bodylen_real * chest_real < 1800:
        #     real_weight = chest_real * 2 * bodylen_real / 138
        # elif bodylen_real * chest_real >= 1800 and bodylen_real * chest_real < 3000:
        #     real_weight = chest_real * 2 * bodylen_real / 120
        # else:
        #     real_weight = chest_real * 2 * bodylen_real / 118
        # print(real_weight)
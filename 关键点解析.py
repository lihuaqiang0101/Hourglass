# import json
# import os
#
# fw = open('两个点2.txt','w')
# for dir in os.listdir(r'D:\MASKpicture\datas\张辉艳\outputs'):
#     try:
#         with open(os.path.join(r'D:\MASKpicture\datas\张辉艳\outputs',dir),encoding='utf8') as f:
#             datas = json.load(f)
#             name = datas['path'].split('\\')[-1]
#             bboxs = datas['outputs']['object']
#             fw.write(name + ' ')
#             for box in bboxs:
#                 if box['name'] == 'e':
#                     e = box['bndbox']
#                 elif box['name'] == 'p1':
#                     p =box['bndbox']
#             fw.write(str(e['xmin']) + ' ' + str(e['ymin']) +' ')
#             fw.write(str(p['xmin']) + ' ' + str(p['ymin']) + '\n')
#     except:
#         pass
# exit()

import cv2
import json
import os

from math import *
def rota(img,degree,W,H,x1,y1,x2,y2):
    if degree == 90:
        tempx = x1
        x1 = y1
        y1 = W - tempx

        tempx = x2
        x2 = y2
        y2 = W - tempx
    elif degree == 180:
        x1 = W - x1
        y1 = H - y1
        x2 = W - x2
        y2 = H - y2
    elif degree == 270:
        tempx = x1
        x1 = H - y1
        y1 = tempx

        tempx = x2
        x2 = H - y2
        y2 = tempx

    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,x1,y1,x2,y2

count = 5192
fw = open('关键点检测.txt','a')
with open(r'rule\babel.txt') as f:
    datas = f.readlines()
    for data in datas:
        data = data.split(' ')
        try:
            image = cv2.imread(os.path.join(r'D:\MASKpicture\datas\datasets\jinglingbiaozhu',data[0]))
            H, W, _ = image.shape
            cv2.imwrite(r'D:\MASKpicture\1_pig\{}.jpg'.format(count),image)
            fw.write(str(count) + '.jpg' + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4])
            count += 1
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[3])
            y2 = int(data[4])
            for rote in [90,180,270]:
                image_,x1_,y1_,x2_,y2_ = rota(image,rote,W,H,x1,y1,x2,y2)
                cv2.imwrite(r'D:\MASKpicture\1_pig\{}.jpg'.format(count),image_)
                fw.write(str(count) + '.jpg' + ' ' + str(x1_) + ' ' + str(y1_) + ' ' + str(x2_) + ' ' + str(y2_) + '\n')
                count += 1
            # cv2.circle(image, (x1, y1), 60,(0, 0, 255), 10)
            # cv2.circle(image, (x2, y2), 60, (0, 0, 255), 10)
            # image = cv2.resize(image,(512,512))
            # cv2.imshow('',image)
            # cv2.waitKey(9)
        except:
            pass
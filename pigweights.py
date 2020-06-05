import torch
from torch import nn
import numpy as np
from PIL import Image
from skimage.transform import resize
import cv2
import torchvision
import torch.hub
import torch.nn.functional as F
import resnet
from collections import OrderedDict
from skimage import color,measure
from collections import Counter
from math import *
import os

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model

def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        pass
    return model

def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)

class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.PReLU()
        )  # 112
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.PReLU()
        )  # 56
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.PReLU()
        )  # 28
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3,stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512,512, 3, padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, 3,padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=0),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(128,128, 3, padding=0),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv12 = nn.Conv2d(128, 5, 3, stride=3, padding=0)
        self.confidence = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = x.view(-1, 5)
        position = x[:, :4]
        confidence = self.confidence(x[:, -1])
        return position, confidence

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.ReLU()
        )  # 112
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.ReLU()
        )  # 56
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.ReLU()
        )  # 28
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.ReLU()
        )  # 14
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.ReLU()
        )  # 16
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.ReLU()
        )  # 8
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3,stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )  # 8
        self.fc = nn.Sequential(
            nn.Linear(256*3*3,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(-1, 256*3*3)
        confidence = self.fc(x)
        return confidence


class NetLL(nn.Module):
    def __init__(self):
        super(NetLL, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(1,64),
            nn.PReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64,128),
            nn.PReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128,256),
            nn.PReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256,128),
            nn.PReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(128,1),
            nn.PReLU()
        )
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)

def counter(arr):
    return Counter(arr).most_common(2)  # 返回出现频率最高的两个数

def get_h(path):
    img1 = cv2.imread(path, 0)
    _, thresh = cv2.threshold(img1, 200, 255, 0)
    image, contours = cv2.findContours(thresh, 2, 1)
    area = 0
    for c in image:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        y_max = np.max(box[:, 1])
        x_max = np.max(box[:, 0])
        y_min = np.min(box[:, 1])
        x_min = np.min(box[:, 0])
        if (y_max - y_min) * (x_max - x_min) > area:
            area = (y_max - y_min) * (x_max - x_min)
            bbox = box
            yy_max = y_max
            xx_max = x_max
            yy_min = y_min
            xx_min = x_min
    # cv2.drawContours(img1, [bbox], 0, (100, 100, 100), 4)
    # cv2.imshow('a', img1)
    # cv2.waitKey(0)
    flag = 0  # 0代表不是最标准的长方形
    if bbox[0][0] == bbox[1][0] or bbox[0][0] == bbox[2][0] or bbox[0][0] == bbox[3][0]:
        flag = 1  # 表示是横屏竖直
    if flag != 1:
        x1y1 = bbox[bbox[:, 0] == xx_min][0]
        x3y3 = bbox[bbox[:, 0] == xx_max][0]
        x4y4 = bbox[bbox[:, 1] == yy_max][0]
        a = x4y4[1] - x1y1[1]
        b = x4y4[0] - x1y1[0]
        c = x3y3[0] - x4y4[0]
        d = x4y4[1] - x3y3[1]
        w = sqrt(a ** 2 + b ** 2) if sqrt(a ** 2 + b ** 2) < sqrt(c ** 2 + d ** 2) else sqrt(c ** 2 + d ** 2)
        h = sqrt(a ** 2 + b ** 2) if sqrt(a ** 2 + b ** 2) > sqrt(c ** 2 + d ** 2) else sqrt(c ** 2 + d ** 2)
    else:
        w = (xx_max - xx_min) if (xx_max - xx_min) < (yy_max - yy_min) else (yy_max - yy_min)
        h = (xx_max - xx_min) if (xx_max - xx_min) > (yy_max - yy_min) else (yy_max - yy_min)
    return h
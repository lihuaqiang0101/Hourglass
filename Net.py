from torch import nn
import torch

class HgResBlock(nn.Module):#输出尺寸和输入尺寸相等
    ''' Hourglass residual block '''

    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1, stride)  # bias=False
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, 3, stride, 1)
        self.bn3 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, 1, stride)  # bias=False
        self.relu = nn.ReLU(inplace=True)
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out

class Hourglass(nn.Module):#输出尺寸和输入尺寸相等
    def __init__(self, depth, nFeat, nModules, resBlock):
        super().__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules  # num residual modules per location
        self.resBlock = resBlock

        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_hour_glass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in
                   range(3)]  # skip(upper branch); down_path, up_path(lower branch)
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))  # extra one for the middle
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def forward(self, x):
        return self._hour_glass_forward(0, x)

    def _hour_glass_forward(self, depth_id, x):
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1)
        low3 = self.hg[depth_id][2](low2)
        up2 = self.upsample(low3)
        return up1 + up2

class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=3):
        super().__init__()
        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.sigmoid = nn.Sigmoid()
        self._make_head()

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(nStacks):
            hg.append(Hourglass(4, nFeat, nModules, resBlock))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, 1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, 1))
                score_.append(nn.Conv2d(nClasses, nFeat, 1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_head(self):
        self.conv1 = nn.Conv2d(self.inplanes, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)
        self.up = nn.Sequential(
            UpsampleLayer()
        )

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True))

    def forward(self, x):
        # head，降采样1倍
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res1(x)#降采样1倍，直到res3都保持相同的尺寸
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)

        out = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(self.sigmoid(score))
            if i < (self.nStacks - 1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out

# net = HourglassNet(nStacks=2, nModules=4, nFeat=128, nClasses=6)
# net.cuda()
# x = torch.rand(2,3,512,512)
# x = x.cuda()
# outputs = net(x)[-1]
# print(outputs.shape)
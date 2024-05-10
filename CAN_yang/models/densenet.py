import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import spconv
import spconv.pytorch as spconv


# DenseNet-B
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class TetrisKernels(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, pool=True):
        super(TetrisKernels, self).__init__()
        self.conv1 = spconv.SparseSequential(spconv.SubMConv2d(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2)),
                                             # swiss()
                                             )
        self.in1 = nn.InstanceNorm2d(out_ch, affine=True)

        self.conv2 = spconv.SparseSequential(spconv.SubMConv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size//2, 0)),
                                             # swiss()
                                             )
        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)

        self.conv3 = spconv.SparseSequential(spconv.SubMConv2d(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2)),
                                             spconv.SubMConv2d(out_ch, out_ch, kernel_size=(kernel_size, 1), stride=1,
                                                               padding=(kernel_size // 2, 0)),
                                             # swiss()
                                             )
        self.in3 = nn.InstanceNorm2d(out_ch, affine=True)

        self.conv4 = spconv.SparseSequential(
            spconv.SubMConv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size // 2, 0)),
            spconv.SubMConv2d(out_ch, out_ch, kernel_size=(1, kernel_size), stride=1,
                              padding=(0, kernel_size // 2)),
            # swiss()
        )
        self.in4 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.MaxPool2d(2, 2)
        self.to_dense = spconv.ToDense()
        self.sig = nn.Sigmoid()
        self.pool = pool
        self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=(7 - 1) // 2, bias=False)

    def forward(self, x):
        x0 = spconv.SparseConvTensor.from_dense(x.permute(0, 2, 3, 1))
        x1 = self.to_dense(self.conv1(x0))
        x1_in = self.in1(x1)
        x2 = self.to_dense(self.conv2(x0))
        x2_in = self.in2(x2)
        x3 = self.to_dense(self.conv3(x0))
        x3_in = self.in3(x3)
        x4 = self.to_dense(self.conv4(x0))
        x4_in = self.in4(x4)

        # x5 = self.conv50(x)
        # x5_in = self.in5(x5)
        # x6 = self.conv6(x)
        # x6_in = self.in6(x6)
        # x7 = self.conv7(x)
        # x7_in = self.in7(x7)
        # x8 = self.conv8(x)
        # x8_in = self.in6(x8)

        global_weights = torch.cat((self.in1.weight.unsqueeze(dim=0), self.in2.weight.unsqueeze(dim=0),
                                    self.in3.weight.unsqueeze(dim=0), self.in4.weight.unsqueeze(dim=0),
                                    # self.in5.weight.unsqueeze(dim=0), self.in6.weight.unsqueeze(dim=0),
                                    # self.in7.weight.unsqueeze(dim=0), self.in8.weight.unsqueeze(dim=0),
                                    ), dim=1).unsqueeze(dim=1)
        global_weights = self.conv(global_weights).squeeze(dim=0)
        # sub_weights = torch.split(global_weights, global_weights.shape[1]//8, dim=1)
        sub_weights = torch.split(global_weights, global_weights.shape[1] // 4, dim=1)
        x1 = self.sig(sub_weights[0].unsqueeze(dim=-1).unsqueeze(dim=-1) * x1_in)
        x2 = self.sig(sub_weights[1].unsqueeze(dim=-1).unsqueeze(dim=-1) * x2_in)
        x3 = self.sig(sub_weights[2].unsqueeze(dim=-1).unsqueeze(dim=-1) * x3_in)
        x4 = self.sig(sub_weights[3].unsqueeze(dim=-1).unsqueeze(dim=-1) * x4_in)
        # x5 = self.sig(sub_weights[4].unsqueeze(dim=-1).unsqueeze(dim=-1) * x5_in)
        # x6 = self.sig(sub_weights[5].unsqueeze(dim=-1).unsqueeze(dim=-1) * x6_in)
        # x7 = self.sig(sub_weights[6].unsqueeze(dim=-1).unsqueeze(dim=-1) * x7_in)
        # x8 = self.sig(sub_weights[7].unsqueeze(dim=-1).unsqueeze(dim=-1) * x8_in)

        # x4 = self.conv4(x)
        # x4 = x1 + x2 + x3 + x4

        x4 = torch.max(torch.max(x1, x2), torch.max(x3, x4))
        # x5 = torch.max(torch.max(x5, x6), torch.max(x7, x8))
        # x4 = torch.max(x4, x5)
        res = x4 * self.conv5(x)
        _, _, h, w = res.shape
        if w % 2 != 0:
            res = torch.nn.functional.pad(res, (0, 1))

        if h % 2 != 0:
            res = torch.nn.functional.pad(res, (0, 0, 0, 1))

        if self.pool:
            res = self.avg_pool(res)
        else:
            res = res
        return res

class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, stride=2, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        # self.layer1_pooling = MOSConvV2(48, 432, 3)
        self.layer2_pooling = TetrisKernels(432, 600, 3)
        self.layer3_pooling = TetrisKernels(600, 684, 3)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out0 = F.relu(out, inplace=True)
        out = F.max_pool2d(out0, 2, ceil_mode=True)
        out1 = self.dense1(out)
        out = self.trans1(out1)
        out2 = self.dense2(out)
        out = self.trans2(out2)
        out3 = self.dense3(out)
        # out1 = out1 + self.layer1_pooling(out0)
        out2 = out2 + self.layer2_pooling(out1)
        out3 = out3 + self.layer3_pooling(out2)
        return out3


# if __name__ == "__main__":
#     from utils import load_config, load_checkpoint, compute_edit_distance
#     from models.mobilenet_v3 import mobilenetv3_large
#     param = load_config(r"D:\ZJU\research\sketch\CAN\config.yaml")
#     x = torch.randn(1, 1, 224, 224)  # torch.Size([1, 684, 14, 14])
#     net = mobilenetv3_large()
#     # net = DenseNet(params=param)
#     out = net(x)
#     print(out.shape)


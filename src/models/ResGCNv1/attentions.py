import torch
from torch import nn


class Part_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_att = self.softmax(self.fcn(x).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Part_Share_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Share_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.part_pool = nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_split = [self.part_pool(x[:,:,:,part]) for part in self.parts]
        x_att = self.softmax(self.fcn(sum(x_split)).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Part_Conv_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Part_Conv_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.part_pool = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ) for _ in range(len(self.parts))])

        self.fcn = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_split = [pool(x[:,:,:,part]) for part, pool in zip(self.parts, self.part_pool)]
        x_att = self.softmax(self.fcn(sum(x_split)).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_att = self.fcn(x).squeeze()
        return self.relu(self.bn(x * x_att[:, :, None, None]) + res)


class Frame_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_avg = torch.transpose(self.avg_pool(torch.transpose(x, 1, 2)), 1, 2)
        x_max = torch.transpose(self.max_pool(torch.transpose(x, 1, 2)), 1, 2)
        x_att = self.conv(torch.cat([x_avg, x_max], dim=1)).squeeze()
        return self.relu(self.bn(x * x_att[:, None, :, None]) + res)


class Joint_Att(nn.Module):
    def __init__(self, channel, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2d(num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_att = self.fcn(torch.transpose(x, 1, 3)).squeeze()
        return self.relu(self.bn(x * x_att[:, None, None, :]) + res)


def get_corr_joints(parts):
    num_joints = max([max(part) for part in parts]) + 1
    res = []
    for i in range(num_joints):
        for j in range(len(parts)):
            if i in parts[j]:
                res.append(j)
                break
    return torch.Tensor(res).long()

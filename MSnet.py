import torch
from torch import nn
from torch.autograd import Variable

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
                module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class SRA(nn.Module):
    def __init__(self, ch_num):
        super(SRA, self).__init__()
        self.conv_mm = nn.Conv3d(ch_num, 1, kernel_size=1, bias=True)
        self.conv_sm = nn.Conv3d(ch_num, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, all, sm_t1c, sm_fl):
        sm_cat = torch.cat([sm_t1c, sm_fl], dim=1)
        ct_mm = self.conv_mm(all)
        ct_sm = self.conv_sm(sm_cat)
        ct = torch.cat([ct_mm, ct_sm], dim=1)
        ctp = self.softmax(ct)
        ctp_mm, ctp_sm = ctp[:, 0:1, :, :, :], ctp[:, 1:2, :, :, :]
        fea_mm = all * ctp_mm
        fea_sm = sm_cat * ctp_sm
        return fea_mm, fea_sm

class CRA(nn.Module):
    def __init__(self, ch_num):
        super(CRA, self).__init__()
        self.ch_num = ch_num
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_mm = nn.Sequential(
            nn.Linear(ch_num * 2, ch_num),
            nn.ReLU(inplace=True),
            nn.Linear(ch_num, ch_num),
            nn.Sigmoid()
        )
        self.fc_sm = nn.Sequential(
            nn.Linear(ch_num * 2, ch_num),
            nn.ReLU(inplace=True),
            nn.Linear(ch_num, ch_num),
            nn.Sigmoid()
        )
    def forward(self, fea_mm, fea_sm):
        fea = torch.cat([fea_mm, fea_sm], dim=1)
        b, c, _, _, _ = fea.size()
        aef = self.avg_pool(fea).view(b, c)
        aef_mm = self.fc_mm(aef).view(b, self.ch_num, 1, 1, 1)
        aef_sm = self.fc_sm(aef).view(b, self.ch_num, 1, 1, 1)
        fea_mmc = aef_mm * fea_mm
        fea_smc = aef_sm * fea_sm
        return fea_mmc+fea_smc


class DCM(nn.Module):
    def __init__(self, ch_num):
        super(DCM, self).__init__()
        self.sra = SRA(ch_num)
        self.cra = CRA(ch_num)
    def forward(self, all, sm_t1c, sm_fl):
        fea_mm, fea_sm = self.sra(all, sm_t1c, sm_fl)
        fea = self.cra(fea_mm, fea_sm)
        return fea


class MM_block1(nn.Module):
    def __init__(self):
        super(MM_block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.down = nn.MaxPool3d(2)

    def forward(self, x):
        x_d = self.conv(x)
        x = self.down(x_d)
        return x, x_d

class MM_block2(nn.Module):
    def __init__(self):
        super(MM_block2, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.dcm = DCM(64)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.down = nn.MaxPool3d(2)

    def forward(self, x, sm_t1c, sm_fl):
        x = self.conv_0(x)
        x = self.dcm(x, sm_t1c, sm_fl)
        x_d = self.conv_1(x)
        x = self.down(x_d)
        return x, x_d

class MM_block3(nn.Module):
    def __init__(self):
        super(MM_block3, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.dcm = DCM(128)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.down = nn.MaxPool3d(2)

    def forward(self, x, sm_t1c, sm_fl):
        x = self.conv_0(x)
        x = self.dcm(x, sm_t1c, sm_fl)
        x_d = self.conv_1(x)
        x = self.down(x_d)
        return x, x_d

class MM_block4(nn.Module):
    def __init__(self):
        super(MM_block4, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.down = nn.MaxPool3d(2)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.up_0 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.conv_2 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.up_1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

    def forward(self, x):
        x_ = self.conv_0(x)
        x = self.down(x_)
        x = self.conv_1(x)
        x = self.up_0(x)
        x = torch.cat([x, x_], dim=1)
        x = self.conv_2(x)
        x = self.up_1(x)
        return x

class MM_block5(nn.Module):
    def __init__(self):
        super(MM_block5, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.dcm = DCM(128)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.up = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)

    def forward(self, x, x_d, sm_t1c, sm_fl):
        x = torch.cat([x, x_d], dim=1)  # b,c,x,y,z
        x = self.conv_0(x)
        x = self.dcm(x, sm_t1c, sm_fl)
        x = self.conv_1(x)
        x = self.up(x)
        return x

class MM_block6(nn.Module):
    def __init__(self):
        super(MM_block6, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.dcm = DCM(64)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.up = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)

    def forward(self, x, x_d, sm_t1c, sm_fl):
        x = torch.cat([x, x_d], dim=1)  # b,c,x,y,z
        x = self.conv_0(x)
        x = self.dcm(x, sm_t1c, sm_fl)
        x = self.conv_1(x)
        x = self.up(x)
        return x

class MM_block7(nn.Module):
    def __init__(self):
        super(MM_block7, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(32, 4, kernel_size=3, padding=1)
        )

    def forward(self, x, x_d):
        x = torch.cat([x, x_d], dim=1)  # b,c,x,y,z
        x = self.conv_0(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(in_channels*2, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class SM_net(nn.Module):
    def __init__(self):
        super(SM_net, self).__init__()
        self.down1 =  nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        self.up1 = Up(64, 64)
        self.up2 = Up(64, 32)
        self.out = Out(32, 4)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        o = self.out(x6, x1)
        return [x2, x3, x5, x6, o]

class MM_net(nn.Module):
    def __init__(self):
        super(MM_net, self).__init__()
        self.mm_block1 = MM_block1()
        self.mm_block2 = MM_block2()
        self.mm_block3 = MM_block3()
        self.mm_block4 = MM_block4()
        self.mm_block5 = MM_block5()
        self.mm_block6 = MM_block6()
        self.mm_block7 = MM_block7()
        self.sm_net_t1c  = SM_net()
        self.sm_net_fl = SM_net()
        self.weightInitializer = InitWeights_He
        self.apply(self.weightInitializer)

    def forward(self, x):
        t1c = x[:, 2:3, :, :, :]
        flair = x[:, 0:1, :, :, :]
        t1c_list = self.sm_net_t1c(t1c)
        flair_list = self.sm_net_fl(flair)
        x, x_d1 = self.mm_block1(x)
        x, x_d2 = self.mm_block2(x, t1c_list[0], flair_list[0])
        x, x_d3 = self.mm_block3(x, t1c_list[1], flair_list[1])
        x = self.mm_block4(x)
        x = self.mm_block5(x, x_d3, t1c_list[2], flair_list[2])
        x = self.mm_block6(x, x_d2, t1c_list[3], flair_list[3])
        out = self.mm_block7(x, x_d1)
        return out, t1c_list[4], flair_list[4]
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2


class Cross_transformer_backbone(nn.Module):
    def __init__(self, in_channels=48):
        super(Cross_transformer_backbone, self).__init__()

        self.to_key = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.to_value = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.cam_layer0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cam_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cam_layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_feature, features):
        Query_features = input_feature
        Query_features = self.cam_layer0(Query_features)
        key_features = self.cam_layer1(features)
        value_features = self.cam_layer2(features)

        QK = torch.einsum("nlhd,nshd->nlsh", Query_features, key_features)
        softmax_temp = 1. / Query_features.size(3) ** .5
        A = torch.softmax(softmax_temp * QK, dim=2)
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, value_features).contiguous()
        message = self.mlp(torch.cat([input_feature, queried_values], dim=1))

        return input_feature + message























class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(NeighborFeatureAggregation, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        # scale 2
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, self.in_d[1], self.out_d)
        # scale 3
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, self.in_d[2], self.out_d)
        # scale 4
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, self.in_d[3], self.out_d)
        # scale 5
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, self.in_d[4], self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        c2_s2 = self.conv_scale2_c2(c2)

        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')

        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], dim=1), c2)
        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)

        c3_s3 = self.conv_scale3_c3(c3)

        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')

        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3)
        # scale 4
        c3_s4 = self.conv_scale4_c3(c3)

        c4_s4 = self.conv_scale4_c4(c4)

        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear')

        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)
        # scale 5
        c4_s5 = self.conv_scale5_c4(c4)

        c5_s5 = self.conv_scale5_c5(c5)

        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        # branch 1
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x1, x2):
        # temporal fusion
        x = torch.abs(x1 - x2)
        # branch 1
        x_branch1 = self.conv_branch1(x)
        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out


class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(TemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # fusion
        self.tffm_x2 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x3 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x4 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x5 = TemporalFeatureFusionModule(self.in_d, self.out_d)

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        # temporal fusion
        c2 = self.tffm_x2(x1_2, x2_2)
        c3 = self.tffm_x3(x1_3, x2_3)
        c4 = self.tffm_x4(x1_4, x2_4)
        c5 = self.tffm_x5(x1_5, x2_5)

        return c2, c3, c4, c5


class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask


class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # high-level
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5

class DS_layer(nn.Module):#拼接DSAMNet的，用来降维
    def __init__(self, in_d, out_d, stride, output_padding):
        super(DS_layer, self).__init__()

        self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)
        self.n_class=1
        n_class=self.n_class
        self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.dsconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x

class DS_layer2(nn.Module):#拼接DSAMNet的，用来降维
    def __init__(self, in_d, out_d, stride, output_padding):
        super(DS_layer2, self).__init__()

        self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)
        self.n_class=64
        n_class=self.n_class
        self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.dsconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x

class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.swa = NeighborFeatureAggregation(channles, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.en_d * 2)
        self.decoder = Decoder(self.en_d * 2)

        # Relation-aware
        self.Cross_transformer_backbone_a3 = Cross_transformer_backbone(in_channels=channles[4])
        self.Cross_transformer_backbone_a2 = Cross_transformer_backbone(in_channels=channles[3])
        self.Cross_transformer_backbone_a1 = Cross_transformer_backbone(in_channels=channles[2])
        self.Cross_transformer_backbone_a0 = Cross_transformer_backbone(in_channels=channles[1])
        self.Cross_transformer_backbone_a33 = Cross_transformer_backbone(in_channels=channles[4])
        self.Cross_transformer_backbone_a22 = Cross_transformer_backbone(in_channels=channles[3])
        self.Cross_transformer_backbone_a11 = Cross_transformer_backbone(in_channels=channles[2])
        self.Cross_transformer_backbone_a00 = Cross_transformer_backbone(in_channels=channles[1])

        self.Cross_transformer_backbone_b3 = Cross_transformer_backbone(in_channels=channles[4])
        self.Cross_transformer_backbone_b2 = Cross_transformer_backbone(in_channels=channles[3])
        self.Cross_transformer_backbone_b1 = Cross_transformer_backbone(in_channels=channles[2])
        self.Cross_transformer_backbone_b0 = Cross_transformer_backbone(in_channels=channles[1])
        self.Cross_transformer_backbone_b33 = Cross_transformer_backbone(in_channels=channles[4])
        self.Cross_transformer_backbone_b22 = Cross_transformer_backbone(in_channels=channles[3])
        self.Cross_transformer_backbone_b11 = Cross_transformer_backbone(in_channels=channles[2])
        self.Cross_transformer_backbone_b00 = Cross_transformer_backbone(in_channels=channles[1])

        self.ds_lyr2 = DS_layer(16, 16, 2, 1)
        self.ds_lyr3 = DS_layer(64, 32, 4, 3)

        self.ds_lj1 = DS_layer2(32, 16, 1, 0)
        self.ds_lj2 = DS_layer2(48, 16, 1, 0)
        self.ds_lj3 = DS_layer2(64, 16, 1, 0)
        self.ds_lj4 = DS_layer2(192, 16, 1, 0)
        self.ds_lj5 = DS_layer2(640, 16, 1, 0)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        #lj=torch.cat((x1_1,x2_1),dim=1)# 2 32 128 128
        lj1=self.ds_lj1(torch.cat((x1_1,x2_1),dim=1))#2 64 128 128
        lj2 = self.ds_lj2(torch.cat((x1_2, x2_2), dim=1))#2 64 64 64
        lj3 = self.ds_lj3(torch.cat((x1_3, x2_3), dim=1))#2 64 32 32
        lj4 = self.ds_lj4(torch.cat((x1_4, x2_4), dim=1))#2 64 16 16
        lj5 = self.ds_lj5(torch.cat((x1_5, x2_5), dim=1))#2 64 8 8
        # print(lj1.shape)
        # print(lj2.shape)
        # print(lj3.shape)
        # print(lj4.shape)
        # print(lj5.shape)

        # print(x1_1.shape)#2 16 128 128
        #print(x2_2.shape)  #2 24 64 64
        # print(x1_3.shape)  #2 32 32 32
        # print(x1_4.shape)   # 2 96 16 16
        # print(x1_5.shape)    # 2 320 8 8
        aaa={'a','b','c','d'}
        features, features11, features22 = [], [], []
        for i in range(len(aaa)):
            if i==0:
                features11.append(self.Cross_transformer_backbone_a00(x1_2,self.Cross_transformer_backbone_a0(x1_2,x2_2)))
                features22.append(self.Cross_transformer_backbone_b00(x2_2,self.Cross_transformer_backbone_b0(x2_2,x1_2)))

            elif i==1:
                features11.append(self.Cross_transformer_backbone_a11(x1_3,self.Cross_transformer_backbone_a1(x1_3,x2_3)))
                features22.append(self.Cross_transformer_backbone_b11(x2_3,self.Cross_transformer_backbone_a1(x2_3,x1_3)))

            elif i==2:
                features11.append(self.Cross_transformer_backbone_a22(x1_4,self.Cross_transformer_backbone_a2(x1_4,x2_4)))
                features22.append(self.Cross_transformer_backbone_b22(x2_4,self.Cross_transformer_backbone_a2(x2_4,x1_4)))
            elif i==3:

                features11.append(self.Cross_transformer_backbone_a33(x1_5,self.Cross_transformer_backbone_a3(x1_5,x2_5)))
                features22.append(self.Cross_transformer_backbone_b33(x2_5,self.Cross_transformer_backbone_a3(x2_5,x1_5)))

        # print(features11[0].shape)  # 2 24 64 64
        # print(features11[1].shape)  # 2 32 32 32
        # print(features11[2].shape)  # 2 96 16 16
        # print(features11[3].shape)  # 2 320 8 8

        # aggregation
        x1_2, x1_3, x1_4, x1_5 = self.swa(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.swa(x2_2, x2_3, x2_4, x2_5)#neiborhood aggre
        # print(x1_2.shape)  # 2 64 64 64
        # print(x1_3.shape)  # 2 64 32 32
        # print(x1_4.shape)  # 2 64 16 16
        # print(x1_5.shape)  # 2 64 8 8
        # temporal fusion
        c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)

        # print(c2.shape) #2 64 64 64
        # print(c3.shape) #2 64 32 32
        # print(c4.shape)#2 64 16 16
        # print(c5.shape)#2 64 8 8
        # fpn
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)#SAM
        # print(c2.shape) 2 64 64 64
        # print(c3.shape) 2 64 32 32
        # print(mask_p2.shape) #2 1 64 64
        # print(mask_p3.shape)  # 2 1 32  32
        # change map

        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)
        # print(mask_p2.shape)  # 2 1 256 256
        # print(mask_p3.shape)#2 1 256 256

        ds2 = self.ds_lyr2(torch.abs(x1_1 - x2_1))#2 1 256 256
        #print(torch.abs(x1_2 - x2_2).shape)
        ds3 = self.ds_lyr3(torch.abs(x1_2 - x2_2))
        ds2=F.interpolate(ds2, scale_factor=(1,1), mode='bilinear')
        ds2=torch.sigmoid(ds2)

        ds3 = F.interpolate(ds3, scale_factor=(1, 1), mode='bilinear')
        ds3 = torch.sigmoid(ds3)
        # ds3 = self.ds_lyr3(torch.abs(x1_2 - x2_2))
        #
        #print(ds3.shape)
        # x1_1_1 = F.interpolate(x1_1, scale_factor=(4, 4), mode='bilinear')


        return mask_p2, mask_p3, mask_p4, mask_p5,ds2,ds3

import numpy as np
from .visualize import vis_feat
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .res2net import res2net50_v1b_26w_4s
from torch import einsum
import math
from einops import rearrange
from . import MobileNetV2
from . import triplet_attention
from .resnet import resnet18
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,build_activation_layer)
#from mmcv.runner import Sequential
from torch.nn import init
from torch.nn.parameter import Parameter

from torch.autograd import Variable
import torchvision

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import init

from einops import rearrange, repeat
#coordiateattention

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out



# relative positional embedding
#haoleattention
def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class HaloAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        block_size,
        halo_size,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values

        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)

        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)

        # derive queries, keys, values

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))

        # scale

        q *= self.scale

        # attention

        sim = einsum('b i d, b j d -> b i j', q, k)

        # add relative positional bias

        sim += self.rel_pos_emb(q)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device = device)
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge and combine heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)

        # merge blocks back to original feature map

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)
        return out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(DeconvBlock, self).__init__()
        """
        反卷积需要有两个padding
        """

        # padding 是反卷积开始的位置， output_padding 将反卷积之后的图像的边缘部分进行填充
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_act=True):
        x = self.deconv(x)
        if is_act:
            x = torch.relu(self.bn(x))
        return x

def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # 输入图的通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    print(torch.max(edge_detect))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

#CRNet
def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(Variable(im))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def functional_conv2d(im):
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            #conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y



class GhostModule(nn.Module):
    def __init__(self, inp):
        super(GhostModule, self).__init__()
        oup=inp
        kernel_size = 1
        ratio = 2
        dw_size = 3
        stride = 1
        relu = True
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features):
        super(GraphConvolution, self).__init__()
        out_features=in_features,
        bias = True
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        s=8
        self.s = s
        self.module_list = nn.ModuleList()

        in_ch_range=torch.Tensor(in_ch)
        in_ch_list = list(in_ch_range.chunk(chunks=self.s, dim=0))

        self.module_list.append(nn.Sequential())
        channel_nums = []
        for i in range(1,len(in_ch_list)):
            if i == 1:
                channels = len(in_ch_list[i])
            else:
                random_tensor = torch.Tensor(channel_nums[i-2])
                _, pre_ch = random_tensor.chunk(chunks=2, dim=0)
                channels= len(pre_ch)+len(in_ch_list[i])
            channel_nums.append(channels)
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]


class ParNetAttention(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y

class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out



class SEAttention(nn.Module):

    def __init__(self, channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class involution(nn.Module):

    def __init__(self,
                 channels,
                 ):
        super(involution, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        stride = 1
        kernel_size = 3
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out
class RFB_modified(nn.Module):
    """ logical semantic relation (LSR) """
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch3 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.conv_cat = basicConv(4*out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class Attention(nn.Module):
    def __init__(self, in_planes, K, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Conv2d(in_planes, K, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if (init_weight):
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return self.sigmoid(att)


class CondConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, K=K, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output

class CoTAttention(nn.Module):

    def __init__(self, dim,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2

class Contrast_Block_Deep(nn.Module):
    """ local-context contrasted (LCC) """

    def __init__(self, planes):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(32)
        d1 = 4
        d2 = 8
        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        involution(self.outplanes),
        self.ca = nn.ModuleList([
            #CoordAtt(self.outplanes, self.outplanes),
            # CoordAtt(self.outplanes, self.outplanes),
            # CoordAtt(self.outplanes, self.outplanes),
            # CoordAtt(self.outplanes, self.outplanes)

            ShuffleAttention(self.outplanes, self.outplanes),
            ShuffleAttention(self.outplanes, self.outplanes),
            ShuffleAttention(self.outplanes, self.outplanes),
            ShuffleAttention(self.outplanes, self.outplanes)

            # ParNetAttention(self.outplanes),
            # ParNetAttention(self.outplanes),
            # ParNetAttention(self.outplanes),
            # ParNetAttention(self.outplanes)

            # involution(self.outplanes),
            # involution(self.outplanes),
            # involution(self.outplanes),
            # involution(self.outplanes)

            # GhostModule(self.outplanes),
            # GhostModule(self.outplanes),
            # GhostModule(self.outplanes),
            # GhostModule(self.outplanes)

            # GraphConvolution(self.outplanes),
            # GraphConvolution(self.outplanes),
            # GraphConvolution(self.outplanes),
            # GraphConvolution(self.outplanes),

            # HSBlock(self.outplanes),
            # HSBlock(self.outplanes),
            # HSBlock(self.outplanes),
            # HSBlock(self.outplanes)

            # CoTAttention(dim=self.outplanes,kernel_size=3),
            # CoTAttention(dim=self.outplanes, kernel_size=3),
            # CoTAttention(dim=self.outplanes, kernel_size=3),
            # CoTAttention(dim=self.outplanes, kernel_size=3)

            # ECAAttention(kernel_size=3),
            # ECAAttention(kernel_size=3),
            # ECAAttention(kernel_size=3),
            # ECAAttention(kernel_size=3)

            # HaloAttention(dim=self.outplanes,block_size=2,halo_size=1),
            # HaloAttention(dim=self.outplanes, block_size=2, halo_size=1),
            # HaloAttention(dim=self.outplanes, block_size=2, halo_size=1),
            # HaloAttention(dim=self.outplanes, block_size=2, halo_size=1)

            # ParNetAttention(self.outplanes),
            # ParNetAttention(self.outplanes),
            # ParNetAttention(self.outplanes),
            # ParNetAttention(self.outplanes)

            # SequentialPolarizedSelfAttention(self.outplanes),
            # SequentialPolarizedSelfAttention(self.outplanes),
            # SequentialPolarizedSelfAttention(self.outplanes),
            # SequentialPolarizedSelfAttention(self.outplanes)

            # SEAttention(self.outplanes, reduction=8),
            # SEAttention(self.outplanes, reduction=8),
            # SEAttention(self.outplanes, reduction=8),
            # SEAttention(self.outplanes, reduction=8)

            # SpatialGroupEnhance(groups=8),
            # SpatialGroupEnhance(groups=8),
            # SpatialGroupEnhance(groups=8),
            # SpatialGroupEnhance(groups=8)

        ])
        in_channels=planes
        out_channels=64

        self.act = nn.GELU()

    def forward(self, x):


        local_1 = self.local_1(x)
        local_1 = self.ca[0](local_1)
        context_1=self.context_1(x)
        context_1=self.ca[1](context_1)

        local_1 = self.bn1(local_1)
        local_1 = self.relu1(local_1)
        ccl_1 = self.bn1(context_1)
        ccl_1 = self.relu1(ccl_1)
        ccl_1=local_1+ccl_1

        #ccl_1=self.ca[0](ccl_1)




        local_2 = self.local_2(x)
        local_2 = self.ca[2](local_2)
        context_2 = self.context_2(x)
        context_2 = self.ca[3](context_2)
        ccl_2 = self.bn2(context_2)
        ccl_2 = self.relu2(ccl_2)
        local_2 = self.bn2(local_2)
        local_2= self.relu2(local_2)
        ccl_2=local_2+context_2
        #ccl_2=self.ca[2](ccl_2)


        out = torch.cat((ccl_1, ccl_2), 1)


        return out


# class Contrast_Block_Deepgai(nn.Module):
#     """ local-context contrasted (LCC) """
#
#     def __init__(self, planes):
#         super(Contrast_Block_Deepgai, self).__init__()
#         self.inplanes = int(planes)
#         self.outplanes = int(32)
#         d1 = 4
#         d2 = 8
#
#         self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=1, padding=0)
#         # self.invo = involution(self.outplanes)
#         self.invo =GhostModule(self.outplanes)
#
#
#         self.local_2 = involution(self.inplanes)
#         self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)
#
#         self.bn1 = nn.BatchNorm2d(self.outplanes)
#         self.bn2 = nn.BatchNorm2d(self.outplanes)
#
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#
#         self.ca = nn.ModuleList([
#             #CoordAtt(self.outplanes, self.outplanes),
#             # CoordAtt(self.outplanes, self.outplanes),
#             # CoordAtt(self.outplanes, self.outplanes),
#             # CoordAtt(self.outplanes, self.outplanes)
#
#             ShuffleAttention(self.outplanes, self.outplanes),
#             ShuffleAttention(self.outplanes, self.outplanes),
#             ShuffleAttention(self.outplanes, self.outplanes),
#             ShuffleAttention(self.outplanes, self.outplanes)
#
#             # ParNetAttention(self.outplanes),
#             # ParNetAttention(self.outplanes),
#             # ParNetAttention(self.outplanes),
#             # ParNetAttention(self.outplanes)
#
#             # involution(self.outplanes),
#             # involution(self.outplanes),
#             # involution(self.outplanes),
#             # involution(self.outplanes)
#
#             # GhostModule(self.outplanes),
#             # GhostModule(self.outplanes),
#             # GhostModule(self.outplanes),
#             # GhostModule(self.outplanes)
#
#             # GraphConvolution(self.outplanes),
#             # GraphConvolution(self.outplanes),
#             # GraphConvolution(self.outplanes),
#             # GraphConvolution(self.outplanes),
#
#             # HSBlock(self.outplanes),
#             # HSBlock(self.outplanes),
#             # HSBlock(self.outplanes),
#             # HSBlock(self.outplanes)
#
#             # CoTAttention(dim=self.outplanes,kernel_size=3),
#             # CoTAttention(dim=self.outplanes, kernel_size=3),
#             # CoTAttention(dim=self.outplanes, kernel_size=3),
#             # CoTAttention(dim=self.outplanes, kernel_size=3)
#
#             # ECAAttention(kernel_size=3),
#             # ECAAttention(kernel_size=3),
#             # ECAAttention(kernel_size=3),
#             # ECAAttention(kernel_size=3)
#
#             # HaloAttention(dim=self.outplanes,block_size=2,halo_size=1),
#             # HaloAttention(dim=self.outplanes, block_size=2, halo_size=1),
#             # HaloAttention(dim=self.outplanes, block_size=2, halo_size=1),
#             # HaloAttention(dim=self.outplanes, block_size=2, halo_size=1)
#
#             # ParNetAttention(self.outplanes),
#             # ParNetAttention(self.outplanes),
#             # ParNetAttention(self.outplanes),
#             # ParNetAttention(self.outplanes)
#
#             # SequentialPolarizedSelfAttention(self.outplanes),
#             # SequentialPolarizedSelfAttention(self.outplanes),
#             # SequentialPolarizedSelfAttention(self.outplanes),
#             # SequentialPolarizedSelfAttention(self.outplanes)
#
#             # SEAttention(self.outplanes, reduction=8),
#             # SEAttention(self.outplanes, reduction=8),
#             # SEAttention(self.outplanes, reduction=8),
#             # SEAttention(self.outplanes, reduction=8)
#
#             # SpatialGroupEnhance(groups=8),
#             # SpatialGroupEnhance(groups=8),
#             # SpatialGroupEnhance(groups=8),
#             # SpatialGroupEnhance(groups=8)
#
#         ])
#         in_channels=planes
#         out_channels=64
#
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         local_1 = self.local_1(x)
#         local_1 = self.invo(local_1)
#         local_1 = self.ca[0](local_1)
#
#         #ccl_1=self.ca[0](ccl_1)
#         ccl_1 = self.bn1(local_1)
#         ccl_1 = self.relu1(ccl_1)
#
#
#
#         local_2 = self.local_1(x)
#         local_2 = self.invo(local_2)
#         local_2 = self.ca[2](local_2)
#
#         #ccl_2=self.ca[2](ccl_2)
#         ccl_2 = self.bn2(local_2)
#         ccl_2 = self.relu2(ccl_2)
#
#         out = torch.cat((ccl_1, ccl_2), 1)
#
#
#         return out
#CRNet
#A2-Nets
class DoubleAtten(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c):
        super(DoubleAtten,self).__init__()
        self.in_c = in_c
        """Convolve the same input feature map to produce three feature maps with the same scale, i.e., A, B, V (as shown in paper).
        """
        self.convA = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c,kernel_size=1)
    def forward(self,input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w) # reshape A
        atten_map = atten_map.view(b, self.in_c, 1, h*w)       # reshape B to generate attention map
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # Multiply the feature map and the attention weight map to generate a global feature descriptor

        v = self.convV(input)
        atten_vectors = F.softmax(v.view(b, self.in_c, h*w), dim=-1) # 生成 attention_vectors
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0,2,1)

        return out.view(b, _, h, w)
class ShuffleAttention(nn.Module):

   def __init__(self, channel,channelout):
       super().__init__()
       G = 8
       self.G=G
       self.channel=channel
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
       self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
       self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
       self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
       self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
       self.sigmoid=nn.Sigmoid()


   def init_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               init.kaiming_normal_(m.weight, mode='fan_out')
               if m.bias is not None:
                   init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               init.constant_(m.weight, 1)
               init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               init.normal_(m.weight, std=0.001)
               if m.bias is not None:
                   init.constant_(m.bias, 0)


   @staticmethod
   def channel_shuffle(x, groups):
       b, c, h, w = x.shape
       x = x.reshape(b, groups, -1, h, w)
       x = x.permute(0, 2, 1, 3, 4)

       # flatten
       x = x.reshape(b, -1, h, w)

       return x

   def forward(self, x):
       b, c, h, w = x.size()
       #group into subfeatures
       x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

       #channel_split
       x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

       #channel attention
       x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
       x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
       x_channel=x_0*self.sigmoid(x_channel)

       #spatial attention
       x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
       x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
       x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

       # concatenate along channel axis
       out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
       out=out.contiguous().view(b,-1,h,w)

       # channel shuffle
       out = self.channel_shuffle(out, 2)
       return out
class SEWeightModule(nn.Module):

   def __init__(self, channels, reduction=16):
       super(SEWeightModule, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
       self.relu = nn.ReLU(inplace=True)
       self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
       self.sigmoid = nn.Sigmoid()

   def forward(self, x1,x2,x3,x4):
       out1 = self.avg_pool(x1)
       out1 = self.fc1(out1)
       out1 = self.relu(out1)
       out1 = self.fc2(out1)
       out1 = self.sigmoid(out1)
       #x1=x1*out1
       out2 = self.avg_pool(x2)
       out2 = self.fc1(out2)
       out2 = self.relu(out2)
       out2 = self.fc2(out2)
       out2 = self.sigmoid(out2)
       #x2=x2*out2
       out3 = self.avg_pool(x3)
       out3 = self.fc1(out3)
       out3 = self.relu(out3)
       out3 = self.fc2(out3)
       out3 = self.sigmoid(out3)
       #x3=x3*out3
       out4 = self.avg_pool(x4)
       out4= self.fc1(out4)
       out4 = self.relu(out4)
       out4 = self.fc2(out4)
       out4= self.sigmoid(out4)
       #x4=x4*out4
       return x1,x2,x3,x4

## 更换nam为深度可分离卷积
class DeepWise_PointWise_Conv(nn.Module):
   def __init__(self, in_ch, out_ch):
       super(DeepWise_PointWise_Conv, self).__init__()
       self.depth_conv = nn.Conv2d(
           in_channels=in_ch,
           out_channels=in_ch,
           kernel_size=3,
           stride=1,
           padding=1,
           groups=in_ch
       )
       self.point_conv = nn.Conv2d(
           in_channels=in_ch,
           out_channels=out_ch,
           kernel_size=1,
           stride=1,
           padding=0,
           groups=1
       )

   def forward(self, input):
           out = self.depth_conv(input)
           out = self.point_conv(out)
           return out
## 更换nam为深度可分离卷积


#m2snet
class CNN1(nn.Module):
   def __init__(self,channel,map_size,pad):
       super(CNN1,self).__init__()
       self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
       self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
       self.pad = pad
       self.norm = nn.BatchNorm2d(channel)
       self.relu = nn.ReLU()

   def forward(self,x):
       out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
       out = self.norm(out)
       out = self.relu(out)
       return out
#
#resnest_block
'''
Bottleneck Block
'''


class GlobalAvgPool2d(nn.Module):
   '''
   global average pooling 2D class
   '''

   def __init__(self):
       super(GlobalAvgPool2d, self).__init__()

   def forward(self, x):
       return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)


class ConvBlock(nn.Module):
   '''
   convolution block class
   convolution 2D -> batch normalization -> ReLU
   '''

   def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
                ):
       super(ConvBlock, self).__init__()

       self.block = nn.Sequential(
           nn.Conv2d(
               in_channels=in_channels,
               out_channels=out_channels,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               bias=False,
           ),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True)
       )

   def forward(self, x):
       x = self.block(x)
       return x


'''
Split Attention
'''


class rSoftMax(nn.Module):
   '''
   (radix-majorize) softmax class
   input is cardinal-major shaped tensor.
   transpose to radix-major
   '''

   def __init__(self,
                groups=1,
                radix=2
                ):
       super(rSoftMax, self).__init__()

       self.groups = groups
       self.radix = radix

   def forward(self, x):
       B = x.size(0)
       # transpose to radix-major
       x = x.view(B, self.groups, self.radix, -1).transpose(1, 2)
       x = F.softmax(x, dim=1)
       x = x.view(B, -1, 1, 1)

       return x


class SplitAttention(nn.Module):
   '''
   split attention class
   '''

   def __init__(self,
                in_channels,
                channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                radix=2,
                reduction_factor=4
                ):
       super(SplitAttention, self).__init__()

       self.radix = radix

       self.radix_conv = nn.Sequential(
           nn.Conv2d(
               in_channels=in_channels,
               out_channels=channels * radix,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               dilation=dilation,
               groups=groups * radix,
               bias=bias
           ),
           nn.BatchNorm2d(channels * radix),
           nn.ReLU(inplace=True)
       )

       inter_channels = max(32, in_channels * radix // reduction_factor)

       self.attention = nn.Sequential(
           nn.Conv2d(
               in_channels=channels,
               out_channels=inter_channels,
               kernel_size=1,
               groups=groups
           ),
           nn.BatchNorm2d(inter_channels),
           nn.ReLU(inplace=True),
           nn.Conv2d(
               in_channels=inter_channels,
               out_channels=channels * radix,
               kernel_size=1,
               groups=groups
           )
       )

       self.rsoftmax = rSoftMax(
           groups=groups,
           radix=radix
       )

   def forward(self, x):
       # NOTE: comments are ugly...

       '''
       input  : |             in_channels               |
       '''

       '''
       radix_conv : |                radix 0            |               radix 1             | ... |                radix r            |
                    | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k | ... | group 0 | group 1 | ... | group k |
       '''
       x = self.radix_conv(x)

       '''
       split :  [ | group 0 | group 1 | ... | group k |,  | group 0 | group 1 | ... | group k |, ... ]
       sum   :  | group 0 | group 1 | ...| group k |
       '''
       B, rC = x.size()[:2]
       splits = torch.split(x, rC // self.radix, dim=1)
       gap = sum(splits)

       '''
       !! becomes cardinal-major !!
       attention : |             group 0              |             group 1              | ... |              group k             |
                   | radix 0 | radix 1| ... | radix r | radix 0 | radix 1| ... | radix r | ... | radix 0 | radix 1| ... | radix r |
       '''
       att_map = self.attention(gap)

       '''
       !! transposed to radix-major in rSoftMax !!
       rsoftmax : same as radix_conv
       '''
       att_map = self.rsoftmax(att_map)

       '''
       split : same as split
       sum : same as sum
       '''
       att_maps = torch.split(att_map, rC // self.radix, dim=1)
       out = sum([att_map * split for att_map, split in zip(att_maps, splits)])

       '''
       output : | group 0 | group 1 | ...| group k |
       concatenated tensors of all groups,
       which split attention is applied
       '''

       return out.contiguous()

class BottleneckBlock(nn.Module):
   '''
   bottleneck block class
   '''
   expansion = 4

   def __init__(self,
                in_channels,
                channels,
                stride=1,
                dilation=1,
                downsample=None,
                radix=2,
                groups=1,
                bottleneck_width=64,
                is_first=False
                ):
       super(BottleneckBlock, self).__init__()
       group_width = int(channels * (bottleneck_width / 64.)) * groups

       layers = [
           ConvBlock(
               in_channels=in_channels,
               out_channels=group_width,
               kernel_size=1,
               stride=1,
               padding=0
           ),
           SplitAttention(
               in_channels=group_width,
               channels=group_width,
               kernel_size=3,
               stride=stride,
               padding=dilation,
               dilation=dilation,
               groups=groups,
               bias=False,
               radix=radix
           )
       ]

       if stride > 1 or is_first:
           layers.append(
               nn.AvgPool2d(
                   kernel_size=3,
                   stride=stride,
                   padding=1
               )
           )

       layers += [
           nn.Conv2d(
               group_width,
               channels * 4,
               kernel_size=1,
               bias=False
           ),
           nn.BatchNorm2d(channels * 4)
       ]

       self.block = nn.Sequential(*layers)
       self.downsample = downsample

   def forward(self, x):
       residual = x
       if self.downsample:
           residual = self.downsample(x)
       out = self.block(x)
       out += residual

       return F.relu(out)
#resnest_block模块机结束

#TFI-GR CEMR模块
class ChannelAttention(nn.Module):
   def __init__(self, in_planes, ratio=16):
       super(ChannelAttention, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.max_pool = nn.AdaptiveMaxPool2d(1)

       self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
       self.relu1 = nn.ReLU()
       self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
       self.sigmoid = nn.Sigmoid()

   def forward(self, x):
       avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
       max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
       out = avg_out + max_out
       return self.sigmoid(out)
#
#TFI-GR融合crossqkv的两个分支特征
class TemporalFeatureInteractionModule(nn.Module):
   def __init__(self, in_d, out_d):
       super(TemporalFeatureInteractionModule, self).__init__()
       self.in_d = in_d
       self.out_d = out_d
       self.conv_sub = nn.Sequential(
           nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(self.in_d),
           nn.ReLU(inplace=True)
       )
       self.conv_diff_enh1 = nn.Sequential(
           nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(self.in_d),
           nn.ReLU(inplace=True)
       )
       self.conv_diff_enh2 = nn.Sequential(
           nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(self.in_d),
           nn.ReLU(inplace=True)
       )
       self.conv_cat = nn.Sequential(
           nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(self.in_d),
           nn.ReLU(inplace=True)
       )
       self.conv_dr = nn.Sequential(
           nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
           nn.BatchNorm2d(self.out_d),
           nn.ReLU(inplace=True)
       )

   def forward(self, x1, x2):
       # difference enhance
       x_sub = self.conv_sub(torch.abs(x1 - x2))
       x1 = self.conv_diff_enh1(x1.mul(x_sub) + x1)
       x2 = self.conv_diff_enh2(x2.mul(x_sub) + x2)
       # fusion
       x_f = torch.cat([x1, x2], dim=1)
       x_f = self.conv_cat(x_f)
       x = x_sub + x_f
       x = self.conv_dr(x)
       return x
#ASPP
class CBAMLayer(nn.Module):
   def __init__(self, channel, reduction=16, spatial_kernel=7):
       super(CBAMLayer, self).__init__()

       # channel attention 压缩H,W为1
       self.max_pool = nn.AdaptiveMaxPool2d(1)
       self.avg_pool = nn.AdaptiveAvgPool2d(1)

       # shared MLP
       self.mlp = nn.Sequential(
           # Conv2d比Linear方便操作
           # nn.Linear(channel, channel // reduction, bias=False)
           nn.Conv2d(channel, channel // reduction, 1, bias=False),
           # inplace=True直接替换，节省内存
           nn.ReLU(inplace=True),
           # nn.Linear(channel // reduction, channel,bias=False)
           nn.Conv2d(channel // reduction, channel, 1, bias=False)
       )

       # spatial attention
       self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                             padding=spatial_kernel // 2, bias=False)
       self.sigmoid = nn.Sigmoid()

   def forward(self, x):
       max_out = self.mlp(self.max_pool(x))
       avg_out = self.mlp(self.avg_pool(x))
       channel_out = self.sigmoid(max_out + avg_out)
       x = channel_out * x

       max_out, _ = torch.max(x, dim=1, keepdim=True)
       avg_out = torch.mean(x, dim=1, keepdim=True)
       spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
       x = spatial_out * x
       return x

class CBAM_ASPP(nn.Module):                       ##加入通道注意力机制
   def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
       super(CBAM_ASPP, self).__init__()
       self.branch1 = nn.Sequential(
           nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
           nn.BatchNorm2d(dim_out, momentum=bn_mom),
           nn.ReLU(inplace=True),
       )
       self.branch2 = nn.Sequential(
           nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
           nn.BatchNorm2d(dim_out, momentum=bn_mom),
           nn.ReLU(inplace=True),
       )
       self.branch3 = nn.Sequential(
           nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
           nn.BatchNorm2d(dim_out, momentum=bn_mom),
           nn.ReLU(inplace=True),
       )
       self.branch4 = nn.Sequential(
           nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
           nn.BatchNorm2d(dim_out, momentum=bn_mom),
           nn.ReLU(inplace=True),
       )
       self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
       self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
       self.branch5_relu = nn.ReLU(inplace=True)

       self.conv_cat = nn.Sequential(
           nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
           nn.BatchNorm2d(dim_out, momentum=bn_mom),
           nn.ReLU(inplace=True),
       )
       # print('dim_in:',dim_in)
       # print('dim_out:',dim_out)
       self.cbam=CBAMLayer(channel=dim_out*5)

   def forward(self, x):
       [b, c, row, col] = x.size()
       conv1x1 = self.branch1(x)
       conv3x3_1 = self.branch2(x)
       conv3x3_2 = self.branch3(x)
       conv3x3_3 = self.branch4(x)
       global_feature = torch.mean(x, 2, True)
       global_feature = torch.mean(global_feature, 3, True)
       global_feature = self.branch5_conv(global_feature)
       global_feature = self.branch5_bn(global_feature)
       global_feature = self.branch5_relu(global_feature)
       global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

       feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
       # print('feature:',feature_cat.shape)
       # 加入cbam注意力机制
       cbamaspp=self.cbam(feature_cat)
       result1=self.conv_cat(cbamaspp)
       return result1

#ASPPjieshu
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
class attention2d(nn.Module):
   def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
       super(attention2d, self).__init__()
       assert temperature%3==1
       self.avgpool = nn.AdaptiveAvgPool2d(1)
       if in_planes!=3:
           hidden_planes = int(in_planes*ratios)+1
       else:
           hidden_planes = K
       self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
       # self.bn = nn.BatchNorm2d(hidden_planes)
       self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
       self.temperature = temperature
       if init_weight:
           self._initialize_weights()


   def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           if isinstance(m ,nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)

   def updata_temperature(self):
       if self.temperature!=1:
           self.temperature -=3
           print('Change temperature to:', str(self.temperature))


   def forward(self, x):
       x = self.avgpool(x)
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x).view(x.size(0), -1)
       return F.softmax(x/self.temperature, 1)
class Dynamic_conv2d(nn.Module):
   def __init__(self, in_planes=64, out_planes=64):
       super(Dynamic_conv2d, self).__init__()
       self.in_planes = 64
       self.out_planes = 64
       self.stride = 1
       self.padding = 1
       self.dilation = 1
       groups = 1
       self.groups=1
       ratio=0.25
       self.ratio=0.25
       kernel_size=3
       self.kernel_size=3
       bias=True
       self.bias=True
       K=4
       self.K=4
       temperature=34
       init_weight=True

       self.attention = attention2d(in_planes, ratio, K, temperature)

       self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
       if bias:
           self.bias = nn.Parameter(torch.Tensor(K, out_planes))
       else:
           self.bias = None
       if init_weight:
           self._initialize_weights()

       #TODO 初始化
   def _initialize_weights(self):
       for i in range(self.K):
           nn.init.kaiming_uniform_(self.weight[i])


   def update_temperature(self):
       self.attention.updata_temperature()

   def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
       # temporal fusion
       x1 = torch.abs(x1_2 - x2_2)
       x2 = torch.abs(x1_3 - x2_3)
       x3 = torch.abs(x1_4 - x2_4)
       x4 = torch.abs(x1_5 - x2_5)
       softmax_attention = self.attention(x1)
       batch_size, in_planes, height, width = x1.size()
       x1 = x1.view(1, -1, height, width)# 变化成一个维度进行组卷积
       weight = self.weight.view(self.K, -1)

       # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
       aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
       if self.bias is not None:
           aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
           output1 = F.conv2d(x1, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, groups=self.groups*batch_size)
       else:
           output1 = F.conv2d(x1, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, groups=self.groups * batch_size)

       output1 = output1.view(batch_size, self.out_planes, output1.size(-2), output1.size(-1))

       softmax_attention = self.attention(x2)
       batch_size, in_planes, height, width = x2.size()
       x2 = x2.view(1, -1, height, width)  # 变化成一个维度进行组卷积
       weight = self.weight.view(self.K, -1)

       # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
       aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                   self.kernel_size)
       if self.bias is not None:
           aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
           output2 = F.conv2d(x2, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
       else:
           output2 = F.conv2d(x2, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

       output2 = output2.view(batch_size, self.out_planes, output2.size(-2), output2.size(-1))

       softmax_attention = self.attention(x3)
       batch_size, in_planes, height, width = x3.size()
       x3 = x3.view(1, -1, height, width)  # 变化成一个维度进行组卷积
       weight = self.weight.view(self.K, -1)

       # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
       aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                   self.kernel_size)
       if self.bias is not None:
           aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
           output3 = F.conv2d(x3, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
       else:
           output3 = F.conv2d(x3, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

       output3 = output3.view(batch_size, self.out_planes, output3.size(-2), output3.size(-1))

       softmax_attention = self.attention(x4)
       batch_size, in_planes, height, width = x4.size()
       x4 = x4.view(1, -1, height, width)  # 变化成一个维度进行组卷积
       weight = self.weight.view(self.K, -1)

       # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
       aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                   self.kernel_size)
       if self.bias is not None:
           aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
           output4 = F.conv2d(x4, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
       else:
           output4 = F.conv2d(x4, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

       output4 = output4.view(batch_size, self.out_planes, output4.size(-2), output4.size(-1))
       return output1,output2,output3,output4

class ljTemporalFusionModulecelibrateconv(nn.Module):
   def __init__(self, in_d=32, out_d=32):
       super(ljTemporalFusionModulecelibrateconv, self).__init__()
       inplanes = 64
       planes = 64
       stride = 1
       padding = 1
       dilation = 1
       groups = 1
       pooling_r = 4
       norm_layer = nn.BatchNorm2d
       self.k2 = nn.Sequential(
           nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
           nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=False),
           norm_layer(planes),
       )
       self.k3 = nn.Sequential(
           nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=False),
           norm_layer(planes),
       )
       self.k4 = nn.Sequential(
           nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=False),
           norm_layer(planes),
       )

   def forward(self, x1,x2,x3,x4):
       # temporal fusion
       identity1 = x1
       out1 = torch.sigmoid(
           torch.add(identity1, F.interpolate(self.k2(x1), identity1.size()[2:])))  # sigmoid(identity + k2)
       out1 = torch.mul(self.k3(x1), out1)  # k3 * sigmoid(identity + k2)
       out1 = self.k4(out1)  # k4

       identity2 = x2
       out2 = torch.sigmoid(
           torch.add(identity2, F.interpolate(self.k2(x2), identity2.size()[2:])))  # sigmoid(identity + k2)
       out2 = torch.mul(self.k3(x2), out2)  # k3 * sigmoid(identity + k2)
       out2 = self.k4(out2)  # k4

       identity3 = x3
       out3 = torch.sigmoid(
           torch.add(identity3, F.interpolate(self.k2(x3), identity3.size()[2:])))  # sigmoid(identity + k2)
       out3= torch.mul(self.k3(x3), out3)  # k3 * sigmoid(identity + k2)
       out3 = self.k4(out3)  # k4

       identity4 = x4
       out4 = torch.sigmoid(
           torch.add(identity4, F.interpolate(self.k2(x4), identity4.size()[2:])))  # sigmoid(identity + k2)
       out4 = torch.mul(self.k3(x4), out4)  # k3 * sigmoid(identity + k2)
       out4 = self.k4(out4)  # k4

       return out1,out2,out3,out4

class TemporalFusionModulecelibrateconv(nn.Module):
   def __init__(self, in_d=32, out_d=32):
       super(TemporalFusionModulecelibrateconv, self).__init__()
       inplanes = 64
       planes = 64
       stride = 1
       padding = 1
       dilation = 1
       groups = 1
       pooling_r = 4
       norm_layer = nn.BatchNorm2d
       self.k2 = nn.Sequential(
           nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
           nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=False),
           norm_layer(planes),
       )
       self.k3 = nn.Sequential(
           nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=False),
           norm_layer(planes),
       )
       self.k4 = nn.Sequential(
           nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=False),
           norm_layer(planes),
       )

   def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
       # temporal fusion
       x1= torch.abs(x1_2-x2_2)
       x2 = torch.abs(x1_3-x2_3)
       x3 = torch.abs(x1_4- x2_4)
       x4 = torch.abs(x1_5- x2_5)
       identity1 = x1
       out1 = torch.sigmoid(
           torch.add(identity1, F.interpolate(self.k2(x1), identity1.size()[2:])))  # sigmoid(identity + k2)
       out1 = torch.mul(self.k3(x1), out1)  # k3 * sigmoid(identity + k2)
       out1 = self.k4(out1)  # k4

       identity2 = x2
       out2 = torch.sigmoid(
           torch.add(identity2, F.interpolate(self.k2(x2), identity2.size()[2:])))  # sigmoid(identity + k2)
       out2 = torch.mul(self.k3(x2), out2)  # k3 * sigmoid(identity + k2)
       out2 = self.k4(out2)  # k4

       identity3 = x3
       out3 = torch.sigmoid(
           torch.add(identity3, F.interpolate(self.k2(x3), identity3.size()[2:])))  # sigmoid(identity + k2)
       out3= torch.mul(self.k3(x3), out3)  # k3 * sigmoid(identity + k2)
       out3 = self.k4(out3)  # k4

       identity4 = x4
       out4 = torch.sigmoid(
           torch.add(identity4, F.interpolate(self.k2(x4), identity4.size()[2:])))  # sigmoid(identity + k2)
       out4 = torch.mul(self.k3(x4), out4)  # k3 * sigmoid(identity + k2)
       out4 = self.k4(out4)  # k4

       return out1,out2,out3,out4
class BasicConv2d(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, padding=0):
       super(BasicConv2d, self).__init__()
       self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
       self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

   def forward(self, x):
       x = self.conv(x)
       x = self.bn(x)
       return F.relu(x, inplace=True)

#ghostconv
import math
def _make_divisible(v, divisor, min_value=None):
   """
   This function is taken from the original tf repo.
   It ensures that all layers have a channel number that is divisible by 8
   It can be seen here:
   https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
   """
   if min_value is None:
       min_value = divisor
   new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
   # Make sure that round down does not go down by more than 10%.
   if new_v < 0.9 * v:
       new_v += divisor
   return new_v


class SELayer(nn.Module):
   def __init__(self, channel, reduction=4):
       super(SELayer, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.fc = nn.Sequential(
               nn.Linear(channel, channel // reduction),
               nn.ReLU(inplace=True),
               nn.Linear(channel // reduction, channel),        )

   def forward(self, x):
       b, c, _, _ = x.size()
       y = self.avg_pool(x).view(b, c)
       y = self.fc(y).view(b, c, 1, 1)
       y = torch.clamp(y, 0, 1)
       return x * y

class conv1dpool(nn.Module):
   def __init__(self,dim):
       super(conv1dpool, self).__init__()
       pools=[3,7,11]
       self.conv0=nn.Conv2d(dim,dim,7,padding=3,groups=dim)
       self.pool1=nn.AvgPool2d(pools[0],stride=1,padding=pools[0]//2,count_include_pad=False)
       self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
       self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
       self.conv4=nn.Conv2d(dim,dim,1)
       self.sigmoid=nn.Sigmoid()

   def forword(self,x):
       u = x.clone(x)
       x_in = self.conv0(x)
       x_1 = self.poll1(x_in)
       x_2 = self.pool2(x_in)
       x_3 = self.pool3(x_in)
       x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
       return x_out + u

def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
   return nn.Sequential(
       nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
       nn.BatchNorm2d(oup),
       nn.ReLU(inplace=True) if relu else nn.Sequential(),
   )

class involutionTemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(involutionTemporalFeatureFusionModule, self).__init__()
        kernel_size=3

        self.kernel_size = kernel_size
        stride=1
        channels=64
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)


    def forward(self, x):


        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        x_out = (weight * out).sum(dim=3).view(b, self.channels, h, w)

        return x_out




class involutionTemporalFusionModule(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(involutionTemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # fusion
        self.tffm_x2 = involutionTemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x3 = involutionTemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x4 = involutionTemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x5 = involutionTemporalFeatureFusionModule(self.in_d, self.out_d)

    def forward(self, x1,x2,x3,x4):
        # temporal fusion
        c2 = self.tffm_x2(x1)
        c3 = self.tffm_x3(x2)
        c4 = self.tffm_x4(x3)
        c5 = self.tffm_x5(x4)

        return c2, c3, c4, c5
class incTemporalFeatureFusionModule(nn.Module):
   def __init__(self, in_d, out_d):
       super(incTemporalFeatureFusionModule, self).__init__()
       self.in_d = in_d
       self.out_d = out_d
       self.relu = nn.ReLU(inplace=True)
       dim=in_d
       pools = [3, 7, 11]
       self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
       self.conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
       self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
       self.conv3 = nn.Conv2d(192, 64, 1, padding=0, groups=dim)
       self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
       self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
       self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
       self.conv4 = nn.Conv2d(dim, dim, 1)
       self.sigmoid = nn.Sigmoid()

       # branch 1
       self.conv_branch1 = nn.Sequential(
           nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
           nn.BatchNorm2d(self.in_d)
       )
       # branch 2
       # self.convdpool = conv1dpool(self.in_d)
       #self.conv_branch2 =conv1dpool(self.in_d)
       self.conv_branch2_f = nn.Sequential(
           nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
           nn.BatchNorm2d(self.in_d)
       )
       # branch 3
       self.conv_branch3 = conv1dpool(self.in_d)
       self.conv_branch3_f = nn.Sequential(
           nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
           nn.BatchNorm2d(self.in_d)
       )
       # branch 4
       self.conv_branch4 = conv1dpool(self.in_d)
       self.conv_branch4_f = nn.Sequential(
           nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
           nn.BatchNorm2d(self.out_d)
       )
       self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)


   def forward(self, x):


       x_branch1 = self.conv_branch1(x)
       # branch 2
       # yy1 = self.convdpool(x)

       u = x.clone()
       x_in1 = self.conv0(x)
       x_11 = self.pool1(x_in1)
       x_21 = self.pool2(x_in1)
       x_31 = self.pool3(x_in1)
       yy1= self.sigmoid(self.conv4(x_in1 + x_11 + x_21 + x_31)) * u

       x_in2 = self.conv1(x)
       x_12 = self.pool1(x_in2)
       x_22 = self.pool2(x_in2)
       x_32 = self.pool3(x_in2)
       yy2 = self.sigmoid(self.conv4(x_in2 + x_12 + x_22 + x_32)) * u

       x_in3 = self.conv2(x)
       x_13 = self.pool1(x_in3)
       x_23 = self.pool2(x_in3)
       x_33 = self.pool3(x_in3)
       yy3 = self.sigmoid(self.conv4(x_in3 + x_13 + x_23 + x_33)) * u

       x_out=torch.cat((yy1,yy2,yy3),dim=1)
       x_out=self.conv3(x_out)
       return x_out


class incTemporalFusionModule(nn.Module):
   def __init__(self, in_d=32, out_d=32):
       super(incTemporalFusionModule, self).__init__()
       self.in_d = in_d
       self.out_d = out_d
       # fusion
       # self.tffm_x2 = incTemporalFeatureFusionModule(self.in_d, self.out_d)
       # self.tffm_x3 = incTemporalFeatureFusionModule(self.in_d, self.out_d)
       self.tffm_x4 = incTemporalFeatureFusionModule(self.in_d, self.out_d)
       self.tffm_x5 = incTemporalFeatureFusionModule(self.in_d, self.out_d)

   def forward(self, zy4,zy5):
       # temporal fusion

       c4 = self.tffm_x4(zy4)
       c5 = self.tffm_x5(zy5)

       return c4, c5
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
       #yy=self.conv_branch2(x)
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

   def forward(self, d2, d3):
       # high-level


       p3, mask_p3 = self.sam_p3(d3)
       p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
       mask_p2 = self.cls(p2)

       return p2, p3, mask_p2, mask_p3


class DS_layer(nn.Module):  # 拼接DSAMNet的，用来降维
   def __init__(self, in_d, out_d, stride, output_padding):
       super(DS_layer, self).__init__()

       self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                        output_padding=output_padding)
       self.bn = nn.BatchNorm2d(out_d)
       self.relu = nn.ReLU(inplace=True)
       self.dropout = nn.Dropout2d(p=0.2)
       self.n_class = 1
       n_class = self.n_class
       self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

   def forward(self, input):
       x = self.dsconv(input)
       x = self.bn(x)
       x = self.relu(x)
       x = self.dropout(x)
       x = self.outconv(x)
       return x


class DS_layer2(nn.Module):  # 拼接DSAMNet的，用来降维
   def __init__(self, in_d, out_d, stride, output_padding):
       super(DS_layer2, self).__init__()

       self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                        output_padding=output_padding)
       self.bn = nn.BatchNorm2d(out_d)
       self.relu = nn.ReLU(inplace=True)
       self.dropout = nn.Dropout2d(p=0.2)
       self.n_class = 64
       n_class = self.n_class
       self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

   def forward(self, input):
       x = self.dsconv(input)
       x = self.bn(x)
       x = self.relu(x)
       x = self.dropout(x)
       x = self.outconv(x)
       return x


class DS_layer3(nn.Module):  # 拼接DSAMNet的，用来降维
   def __init__(self, in_d, out_d, stride, output_padding):
       super(DS_layer3, self).__init__()

       self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                        output_padding=output_padding)
       self.bn = nn.BatchNorm2d(out_d)
       self.relu = nn.ReLU(inplace=True)
       self.dropout = nn.Dropout2d(p=0.2)

       self.outconv = nn.ConvTranspose2d(out_d, out_d, kernel_size=3, padding=1)

   def forward(self, input):
       x = self.dsconv(input)
       x = self.bn(x)
       x = self.relu(x)
       x = self.dropout(x)
       x = self.outconv(x)
       return x

class rednestDS_layer3(nn.Module):  # 拼接resnest的，用来降维
   def __init__(self, in_d, out_d, stride, output_padding):
       super(rednestDS_layer3, self).__init__()

       self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                        output_padding=output_padding)
       self.bn = nn.BatchNorm2d(out_d)
       self.relu = nn.ReLU(inplace=True)
       self.dropout = nn.Dropout2d(p=0.2)

       self.outconv = nn.ConvTranspose2d(out_d, out_d, kernel_size=3, padding=1)

   def forward(self, input):
       x = self.dsconv(input)
       x = self.bn(x)
       x = self.relu(x)
       x = self.dropout(x)
       x = self.outconv(x)
       return x

class Conv(nn.Module):
   def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
       super(Conv, self).__init__()
       self.inp_dim = inp_dim
       self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
       self.relu = None
       self.bn = None
       if relu:
           self.relu = nn.ReLU(inplace=True)
       if bn:
           self.bn = nn.BatchNorm2d(out_dim)

   def forward(self, x):
       assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
       # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
       x = self.conv(x)
       if self.bn is not None:
           x = self.bn(x)
       if self.relu is not None:
           x = self.relu(x)
       return x


class Residual(nn.Module):
   def __init__(self, inp_dim, out_dim):
       super(Residual, self).__init__()
       self.relu = nn.ReLU(inplace=True)
       self.bn1 = nn.BatchNorm2d(inp_dim)
       self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
       self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
       self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
       self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
       self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
       self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
       if inp_dim == out_dim:
           self.need_skip = False
       else:
           self.need_skip = True

   def forward(self, x):
       if self.need_skip:
           residual = self.skip_layer(x)
       else:
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
       out += residual
       return out


class ConvRelPosEnc(nn.Module):
   """ Convolutional relative position encoding. """

   def __init__(self, Ch, h, window):
       """
       Initialization.
           Ch: Channels per head.
           h: Number of heads.
           window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                   1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                   2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                      It will apply different window size to the attention head splits.
       """
       super().__init__()

       if isinstance(window, int):
           window = {window: h}  # Set the same window size for all attention heads.
           self.window = window
       elif isinstance(window, dict):
           self.window = window
       else:
           raise ValueError()

       self.conv_list = nn.ModuleList()
       self.head_splits = []
       for cur_window, cur_head_split in window.items():
           dilation = 1  # Use dilation=1 at default.
           padding_size = (cur_window + (cur_window - 1) * (
                   dilation - 1)) // 2  # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
           cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch,
                                kernel_size=(cur_window, cur_window),
                                padding=(padding_size, padding_size),
                                dilation=(dilation, dilation),
                                groups=cur_head_split * Ch,
                                )
           self.conv_list.append(cur_conv)
           self.head_splits.append(cur_head_split)
       self.channel_splits = [x * Ch for x in self.head_splits]

   def forward(self, q, v, size):
       B, h, N, Ch = q.shape
       H, W = size
       assert N == H * W
       # print(q.shape,v.shape)
       # Convolutional relative position encoding.
       # q_img = q                                                             # Shape: [B, h, H*W, Ch].
       # v_img = v                                                             # Shape: [B, h, H*W, Ch].
       # print(q.shape,v.shape)
       v_img = rearrange(v, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)  # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
       v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels.
       conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
       conv_v_img = torch.cat(conv_v_img_list, dim=1)
       conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch',
                              h=h)  # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

       EV_hat_img = q * conv_v_img
       # print(EV_hat_img.shape)
       zero = torch.zeros((B, h, 0, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
       EV_hat = torch.cat((zero, EV_hat_img), dim=2)  # Shape: [B, h, N, Ch].
       # print(EV_hat.shape)
       return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
   """ Factorized attention with convolutional relative position encoding class. """

   def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
       super().__init__()
       self.num_heads = num_heads
       head_dim = dim // num_heads
       self.scale = head_dim ** -0.5

       self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Note: attn_drop is actually not used.
       self.proj = nn.Linear(dim, dim)
       self.proj_drop = nn.Dropout(proj_drop)

       # Shared convolutional relative position encoding.
       self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window={3: 2, 5: 3, 7: 3})

   def forward(self, q, k, v, size):
       B, N, C = size[0], size[1], size[2]

       # # Generate Q, K, V.
       # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
       # q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].

       # Factorized attention.
       k_softmax = k.softmax(dim=2)  # Softmax on dim N.
       k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)  # Shape: [B, h, Ch, Ch].
       factor_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

       # Convolutional relative position encoding.
       crpe = self.crpe(q, v, size=[size[3], size[4]])  # Shape: [B, h, N, Ch].

       # Merge and reshape.
       x = self.scale * factor_att + crpe
       x = x.transpose(1, 2).reshape(B, N, C)  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

       # Output projection.
       x = self.proj(x)
       x = self.proj_drop(x)

       return x


class PositionalEncoding2D(nn.Module):
   def __init__(self, channels):
       """
       :param channels: The last dimension of the tensor you want to apply pos emb to.
       """
       super(PositionalEncoding2D, self).__init__()
       channels = int(np.ceil(channels / 2))
       self.channels = channels
       inv_freq = 1. / (10000
                        ** (torch.arange(0, channels, 2).float() / channels))
       self.register_buffer('inv_freq', inv_freq)

   def forward(self, tensor):
       """
       :param tensor: A 4d tensor of size (batch_size, x, y, ch)
       :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
       """
       if len(tensor.shape) != 4:
           raise RuntimeError("The input tensor has to be 4d!")
       batch_size, x, y, orig_ch = tensor.shape
       pos_x = torch.arange(x,
                            device=tensor.device).type(self.inv_freq.type())
       pos_y = torch.arange(y,
                            device=tensor.device).type(self.inv_freq.type())
       sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
       sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

       emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                         dim=-1).unsqueeze(1)
       emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
       emb = torch.zeros((x, y, self.channels * 2),
                         device=tensor.device).type(tensor.type())
       emb[:, :, :self.channels] = emb_x
       emb[:, :, self.channels:2 * self.channels] = emb_y

       return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
   def __init__(self, channels):
       """
       Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
       """
       super(PositionalEncodingPermute2D, self).__init__()
       self.penc = PositionalEncoding2D(channels)

   def forward(self, tensor):
       tensor = tensor.permute(0, 2, 3, 1)
       enc = self.penc(tensor)
       return enc.permute(0, 3, 1, 2)


class MultiHeadDense(nn.Module):
   def __init__(self, d, bias=False):
       super(MultiHeadDense, self).__init__()
       self.weight = nn.Parameter(torch.Tensor(d, d))
       if bias:
           raise NotImplementedError()
           self.bias = Parameter(torch.Tensor(d, d))
       else:
           self.register_parameter('bias', None)
       self.reset_parameters()

   def reset_parameters(self) -> None:
       nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
       if self.bias is not None:
           fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
           bound = 1 / math.sqrt(fan_in)
           nn.init.uniform_(self.bias, -bound, bound)

   def forward(self, x):
       # x:[b, h*w, d]
       b, wh, d = x.size()
       x = torch.bmm(x, self.weight.repeat(b, 1, 1))
       # x = F.linear(x, self.weight, self.bias)
       return x


class conv2d(nn.Module):
    # use_bn = True,use_rl = True控制是否使用批量归一化和激活函数
    # nn.Conv2d二维卷积方法，参数：
    # groups参数：决定了是否采用分组卷积，现在用的比较多的是groups = in_channel。
    # 当groups = in_channel时，是在做的depth-wise conv的，具体思想可以参考MobileNet那篇论文。
    def __init__(self, in_dim, out_dim, k, pad, stride, groups=1, bias=False, use_bn=True, use_rl=True):
        super(conv2d, self).__init__()
        self.use_bn = use_bn
        self.use_rl = use_rl
        self.conv = nn.Conv2d(in_dim, out_dim, k, padding=pad, stride=stride, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, bottom):
        if self.use_bn and self.use_rl:
            return self.relu(self.bn(self.conv(bottom)))
        elif self.use_bn:
            return self.bn(self.conv(bottom))
        else:
            return self.conv(bottom)

# class BGA(nn.Module):
#     def __init__(self,in_dim):
#         super(BGA,self).__init__()
#         self.in_dim = in_dim
#         self.db_dwconv = nn.conv2d(in_dim,in_dim,3,1,1,in_dim,use_rl=False)
#         self.db_conv1x1 = nn.conv2d(in_dim,in_dim,1,0,1,use_rl=False,use_bn=False)
#         self.db_conv = nn.conv2d(in_dim,in_dim,3,1,2,use_rl=False)
#         self.db_apooling = nn.AvgPool2d(3,2,1)
#
#         self.sb_dwconv = nn.conv2d(in_dim,in_dim,3,1,1,in_dim,use_rl=False)
#         self.sb_conv1x1 = nn.conv2d(in_dim,in_dim,1,0,1,use_rl=False,use_bn=False)
#         self.sb_conv = nn.conv2d(in_dim,in_dim,3,1,1,use_rl=False)
#         self.sb_sigmoid = nn.Sigmoid()
#
#         self.conv = nn.conv2d(in_dim,in_dim,3,1,1,use_rl=False)
#     def forward(self,db1,db2,sb1,sb2):
#         db_dwc = self.db_dwconv(db1)
#         db_out = self.db_conv1x1(db_dwc)#
#         db_conv = self.db_conv(db2)
#         db_pool = self.db_apooling(db_conv)
#
#         sb_dwc = self.sb_dwconv(sb1)
#         sb_out = self.sb_sigmoid(self.sb_conv1x1(sb_dwc))#
#         sb_conv = self.sb_conv(sb2)
#         sb_up = self.sb_sigmoid(F.interpolate(sb_conv, size=db_out.size()[2:], mode="bilinear",align_corners=True))
#
#         return db_out,db_pool,sb_dwc ,sb_up
class enh(nn.Module):
    def __init__(self,in_dim):
        super(enh,self).__init__()
        self.in_dim = in_dim
        self.db_dwconv = conv2d(in_dim, in_dim, 3, 1, 1, in_dim, use_rl=False)
        self.db_conv1x1 = conv2d(in_dim, in_dim, 1, 0, 1, use_rl=False, use_bn=False)


    def forward(self,db1):
        db_dwc = self.db_dwconv(db1)
        db_out = self.db_conv1x1(db_dwc)  #
        return db_out

class adjdb(nn.Module):
    def __init__(self,in_dim):
        super(adjdb ,self).__init__()
        self.in_dim = in_dim
        self.db_conv = conv2d(in_dim, in_dim, 3, 1, 2, use_rl=False)
        self.db_apooling = nn.AvgPool2d(3, 2, 1)


    def forward(self,db):
        db_conv = self.db_conv(db)
        db_pool = self.db_apooling(db_conv)
        return db_pool

class adjsb(nn.Module):
    def __init__(self,in_dim):
        super(adjsb ,self).__init__()
        self.in_dim = in_dim
        self.sb_conv = conv2d(in_dim, in_dim, 3, 1, 1, use_rl=False)
        self.sb_sigmoid = nn.Sigmoid()


    def forward(self,sb):
        sb_conv = self.sb_conv(sb)
        sb_up = self.sb_sigmoid(F.interpolate(sb_conv, size=(32,32), mode="bilinear", align_corners=True))

        return sb_up
class MultiHeadAttention(nn.Module):
   def __init__(self):
       super(MultiHeadAttention, self).__init__()

   def positional_encoding_2d(self, d_model, height, width):
       """
       reference: wzlxjtu/PositionalEncoding2D

       :param d_model: dimension of the model
       :param height: height of the positions
       :param width: width of the positions
       :return: d_model*height*width position matrix
       """
       if d_model % 4 != 0:
           raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
       pe = torch.zeros(d_model, height, width)
       try:
           pe = pe.to(torch.device("cuda:0"))
       except RuntimeError:
           pass
       # Each dimension use half of d_model
       d_model = int(d_model / 2)
       div_term = torch.exp(
           torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
       pos_w = torch.arange(0., width).unsqueeze(1)
       pos_h = torch.arange(0., height).unsqueeze(1)
       pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
           0, 1).unsqueeze(1).repeat(1, height, 1)
       pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
           0, 1).unsqueeze(1).repeat(1, height, 1)
       pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
           0, 1).unsqueeze(2).repeat(1, 1, width)
       pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
           0, 1).unsqueeze(2).repeat(1, 1, width)
       return pe

   def forward(self, x):
       raise NotImplementedError()


class MultiHeadCrossAttention(MultiHeadAttention):
   def __init__(self, channelY, channelS, ch_out, drop_rate=0.2, qkv_bias=False):
       super(MultiHeadCrossAttention, self).__init__()
       self.Sconv = nn.Sequential(
           nn.Conv2d(channelS, channelS, kernel_size=1),
           nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
       self.Yconv = nn.Sequential(
           nn.Conv2d(channelY, channelS, kernel_size=1),
           nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))

       self.query = MultiHeadDense(channelS, bias=False)
       self.key = MultiHeadDense(channelS, bias=False)
       self.value = MultiHeadDense(channelS, bias=False)

       self.softmax = nn.Softmax(dim=1)
       self.Spe = PositionalEncodingPermute2D(channelS)
       self.Ype = PositionalEncodingPermute2D(channelY)

       self.qkv = nn.Linear(channelS, channelS * 3, bias=qkv_bias)
       self.num_heads = 8
       head_dim = channelS // 8
       self.scale = head_dim ** -0.5
       self.factoratt_crpe = FactorAtt_ConvRelPosEnc(channelS, self.num_heads, qkv_bias=qkv_bias, proj_drop=drop_rate)
       self.residual = Residual(channelS * 2, ch_out)
       self.dropout = nn.Dropout2d(drop_rate)
       self.drop_rate = drop_rate

   def forward(self, Y, S):
       Sb, Sc, Sh, Sw = S.size()
       Yb, Yc, Yh, Yw = Y.size()

       Spe = self.Spe(S)
       S = S + Spe
       S1 = self.Sconv(S)
       S1 = S1.reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

       Ype = self.Ype(Y)
       Y = Y + Ype
       Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

       B, N, C = Y1.shape
       size = [B, N, C, Sh, Sw]

       qkv_l = self.qkv(Y1)
       qkv_l = qkv_l.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                   4)  # Shape: [3, B, h, N, Ch].
       q_l, k_l, v_l = qkv_l[0], qkv_l[1], qkv_l[2]

       qkv_g = self.qkv(S1)
       qkv_g = qkv_g.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                   4)  # Shape: [3, B, h, N, Ch].
       q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2]

       cur1 = self.factoratt_crpe(q_g, k_l, v_l, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
       cur2 = self.factoratt_crpe(q_l, k_g, v_g, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

       fuse = self.residual(torch.cat([cur1, cur2], 1))

       cury1 = self.factoratt_crpe(k_l, v_l, q_g, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
       cury2 = self.factoratt_crpe(v_g, k_g, q_l, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

       fusey = self.residual(torch.cat([cury1, cury2], 1))

       fuse=self.dropout(fuse)
       fusey = self.dropout(fusey)
       out=torch.add(fuse,fusey)
       return out, cur1, cur2


class BaseNet(nn.Module):
   def __init__(self, input_nc=3, output_nc=1):
       super(BaseNet, self).__init__()
       self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
       self.tripbletatten = triplet_attention.TripletAttention(gate_channels=64)
       self.tripbletriplet = triplet_attention.TripletAttentionlky(gate_channels=64)
       self.tripbletripletsange = triplet_attention.TripletAttentionsange(gate_channels=64)
       #
       self.DoubleAttent=DoubleAtten(in_c=64)
       self.resnet = resnet18()
       self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'), False)
       self.resnet.layer4 = nn.Identity()

       channles = [16, 24, 32, 96, 320]
       self.en_d = 32
       self.mid_d = self.en_d * 2
       self.swa = NeighborFeatureAggregation(channles, self.mid_d)
       self.tfm = TemporalFusionModule(self.mid_d, self.en_d * 2)
       self.inctfm=incTemporalFusionModule(self.mid_d, self.en_d * 2)
       self.inctfminvolution = involutionTemporalFusionModule(self.mid_d, self.en_d * 2)
       self.ljtfm=TemporalFusionModulecelibrateconv(self.mid_d, self.en_d * 2)
       #self.ljtfm = ljTemporalFusionModulecelibrateconv(self.mid_d, self.en_d * 2)
       self.dynamictfm=Dynamic_conv2d(self.mid_d, self.en_d * 2)
       self.decoder = Decoder(self.en_d * 2)
       self.mid_d = 64
       self.TFIM = TemporalFeatureInteractionModule(64, self.mid_d)
       #TFI-GR  CIEM模块
       self.ca = ChannelAttention(self.mid_d * 4, ratio=16)
       self.conv_dr = nn.Sequential(
           nn.Conv2d(self.mid_d * 4, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
           nn.BatchNorm2d(self.mid_d),
           nn.ReLU(inplace=True)
       )
       # linknetdecoder
       self.decode4 = DeconvBlock(64, 64)
       self.decode3 = DeconvBlock(32, 16)
       self.decode2 = DeconvBlock(16, 8)
       self.decode1 = DeconvBlock(64, 64)
       # 输出部分,第一层走默认即可
       self.deconv_out1 = DeconvBlock(64, 32)
       self.conv_out = ConvBlock(32, 32,3,2,1)
       # stride 为2 可以不写， 一共就是2分类。kesize=2，因为论文给的是2x2的,2x2的适合 padding是不需要变化的，都是0 保证正好变为原来的2倍，因为stride正好是2
       self.deconv_out2 = DeconvBlock(32, 1, k_size=2, padding=0, output_padding=0)
       self.deconv_out3 = DeconvBlock(1, 1, k_size=2, padding=0, output_padding=0)

## 更换nam为resmest
       self.ResNeSt_Block1=BottleneckBlock(24,6)
       self.ResNeSt_Block2 = BottleneckBlock(32, 8)
       self.ResNeSt_Block3 = BottleneckBlock(96, 24)
       self.ResNeSt_Block4 = BottleneckBlock(320, 80)
## 更换nam为resmest
       self.seweight=SEWeightModule(64)
#shufflesttention
       self.shuffleattention=ShuffleAttention(channel=64,channelout=64)
## 更换nam为m2snet
       self.conv_3 = CNN1(64, 3, 1)
       self.conv_5 = CNN1(64, 5, 2)
       self.x5_dem_1 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
       self.x4_dem_1 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
       self.x3_dem_1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
       self.x2_dem_1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
       self.x1_dem_1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

       self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
       self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
       self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
       self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
# 更换nam为m2snet

## 更换nam为深度可分离卷积
       self.depthsplit_conv1 = DeepWise_PointWise_Conv(16, 64)
       self.depthsplit_conv2=DeepWise_PointWise_Conv(24,64)
       self.depthsplit_conv3 = DeepWise_PointWise_Conv(32, 64)
       self.depthsplit_conv4 = DeepWise_PointWise_Conv(96, 64)
       self.depthsplit_conv5 = DeepWise_PointWise_Conv(320, 64)

## 更换nam为深度可分离卷积

##更换nam为MSPSNET
       self.conv6_4_1 = nn.Conv2d(128, 64, padding=1, kernel_size=3, groups=64 // 2,
                                  dilation=1)
       self.conv6_4_2 = nn.Conv2d(128, 64, padding=2, kernel_size=3, groups=64 // 2,
                                  dilation=2)
       self.conv6_4_3 = nn.Conv2d(128, 64, padding=3, kernel_size=3, groups=64 // 2,
                                  dilation=3)
       self.conv6_4_4 = nn.Conv2d(128, 64, padding=4, kernel_size=3, groups=64// 2,
                                  dilation=4)
       self.conv4_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)

       self.conv6_3_1 = nn.Conv2d(128, 64, padding=1, kernel_size=3, groups=64 // 2,
                                  dilation=1)
       self.conv6_3_2 = nn.Conv2d(128, 64, padding=2, kernel_size=3, groups=64// 2,
                                  dilation=2)
       self.conv6_3_3 = nn.Conv2d(128, 64,  padding=3, kernel_size=3, groups=64 // 2,
                                  dilation=3)
       self.conv6_3_4 = nn.Conv2d(128, 64, padding=4, kernel_size=3, groups=64 // 2,
                                  dilation=4)
       self.conv3_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)

       self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4, padding=0)
       self.deconv3 = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=8, padding=0)
       self.deconv4 = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=16, padding=0)
       self.deconv5 = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=32, padding=0)
##更换nam为MSPSNET
       # #ASPP
       # self.CBAM_ASPP1=CBAM_ASPP(16,16)
       # self.CBAM_ASPP2= CBAM_ASPP(24, 24)
       # self.CBAM_ASPP3 = CBAM_ASPP(32, 32)
       # self.CBAM_ASPP4 = CBAM_ASPP(96, 96)
       # self.CBAM_ASPP5 = CBAM_ASPP(320, 320)
       #FHD融和
       # local attention
       # channelss=64
       # inter_channels=int(channelss//4)
       # norm_cfg=dict(type='BN', requires_grad=True)
       # act_cfg = dict(type='ReLU')
       # la_conv1 = ConvModule(channelss, inter_channels, kernel_size=1, stride=1, padding=0)
       # la_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
       # la_act1 = build_activation_layer(act_cfg)
       # la_conv2 = ConvModule(inter_channels, channelss, kernel_size=1, stride=1, padding=0)
       # la_bn2 = build_norm_layer(norm_cfg, channelss)[1]
       # la_layers = [la_conv1, la_bn1, la_act1, la_conv2, la_bn2]
       # self.la_layers = Sequential(*la_layers)
       # # globla attention
       # aap = nn.AdaptiveAvgPool2d(1)
       # ga_conv1 = ConvModule(channelss, inter_channels, kernel_size=1, stride=1, padding=0)
       # ga_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
       # ga_act1 = build_activation_layer(act_cfg)
       # ga_conv2 = ConvModule(inter_channels, channelss, kernel_size=1, stride=1, padding=0)
       # ga_bn2 = build_norm_layer(norm_cfg, channelss)[1]
       # ga_layers = [aap, ga_conv1, ga_bn1, ga_act1, ga_conv2, ga_bn2]
       # self.ga_layers = Sequential(*ga_layers)
       #FHD融合结束
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
       # Generate change map
       self.conv_fusion = nn.Sequential(
           nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False),
           nn.BatchNorm2d(1),
           nn.ReLU(inplace=True)
       )
       self.ds_lyr2 = DS_layer(16, 16, 2, 1)
       self.ds_lyr3 = DS_layer(64, 32, 4, 3)
       self.ds_lyr4 = DS_layer(64, 32,1,0)
       self.ds_lyr41 = DS_layer(4, 1, 1, 0)
       self.ds_lyr5 = DS_layer3(128, 64, 1, 0)
       self.ds_lyr6 = DS_layer3(2, 1, 1, 0)
#量特征backbone后相减支路
       self.ds_lj11 = DS_layer2(16, 16, 1, 0)
       self.ds_lj22 = DS_layer2(24, 16, 1, 0)
       self.ds_lj33 = DS_layer2(32, 16, 1, 0)
       self.ds_lj44 = DS_layer2(96, 16, 1, 0)
       self.ds_lj55 = DS_layer2(320, 16, 1, 0)
# 量特征backbone后相减支路
       self.rednestds_lyr1= rednestDS_layer3(24, 64, 1, 0)
       self.rednestds_lyr2 = rednestDS_layer3(32, 64, 1, 0)
       self.rednestds_lyr3 = rednestDS_layer3(96, 64, 1, 0)
       self.rednestds_lyr4 = rednestDS_layer3(320, 64, 1, 0)
       # 原mobileVITbackbone特征
       self.ds_lj1 = DS_layer2(32, 16, 1, 0)
       self.ds_lj2 = DS_layer2(48, 16, 1, 0)
       self.ds_lj3 = DS_layer2(64, 16, 1, 0)
       self.ds_lj4 = DS_layer2(192, 16, 1, 0)
       self.ds_lj5 = DS_layer2(640, 16, 1, 0)
       # triple通路降为
       self.ljds_lj1 = DS_layer3(32, 16, 1, 0)
       self.ljds_lj2 = DS_layer3(48, 24, 1, 0)
       self.ljds_lj3 = DS_layer3(64, 32, 1, 0)
       self.ljds_lj4 = DS_layer3(192, 96, 1, 0)
       self.ljds_lj5 = DS_layer3(640, 320, 1, 0)
       # resnet18backbone特征降为
       self.dsres_lj1 = DS_layer3(64, 16, 1, 0)
       self.dsres_lj2 = DS_layer3(64, 24, 1, 0)
       self.dsres_lj3 = DS_layer3(128, 32, 1, 0)
       self.dsres_lj4 = DS_layer3(256, 96, 1, 0)
       self.dsres_lj5 = DS_layer3(256, 320, 1, 0)

       self.yibands_lj = DS_layer2(32, 16, 1, 0)
       drop_rate = 0.2
       self.cross2 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate / 2, qkv_bias=True)
       self.cross3 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate / 2, qkv_bias=True)
       self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate / 2, qkv_bias=True)
       self.cross5 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate / 2, qkv_bias=True)
       self.LCC2=Contrast_Block_Deep(24)
       self.LCC3 = Contrast_Block_Deep(32)
       self.LCC4 = Contrast_Block_Deep(96)
       self.LCC5 = Contrast_Block_Deep(320)
       self.enhan1=enh(64)
       self.adjdb=adjdb(64)
       self.adjsb =adjsb(64)

       # self.LCC2 = Contrast_Block_Deepgai(24)
       # self.LCC3 = Contrast_Block_Deepgai(32)
       # self.LCC4 = Contrast_Block_Deepgai(96)
       # self.LCC5 = Contrast_Block_Deepgai(320)

       self.LSR4 = RFB_modified(96,64)
       self.LSR5 = RFB_modified(320,64)

       self.haole=HaloAttention(dim=64,
        block_size=2,
        halo_size=1,)
       #self.s2att = S2Attention(channels=64)
       self.coord=CoordAtt(inp=64,oup=64)
   def forward(self, x1, x2):
       # edge_detect = edge_conv2d(x1)
       # x1 = np.transpose(edge_detect, (1, 2, 0))
       # slrtempresnet18
       #  c0 = self.resnet.conv1(x1)
       #  c0 = self.resnet.bn1(c0)
       #  c0 = self.resnet.relu(c0)#2 64 128 128
       #  c1 = self.resnet.maxpool(c0)
       #  c1 = self.resnet.layer1(c1)#2 64 64 64
       #  # c1 = self.drop(c1)
       #  c2 = self.resnet.layer2(c1)#2 128 32 32
       #  # c2 = self.drop(c2)
       #  c3 = self.resnet.layer3(c2)#2 256 32 32
       #  c3=self.resnet.maxpool(c3)#2 256 16 16
       # #c3 = self.drop(c3)
       #  c4 = self.resnet.layer4(c3)#2 256 32 32
       #  c4= self.resnet.maxpool(c4)# 2 256 8 8
       #
       #  c0_img2 = self.resnet.conv1(x2)
       #  c0_img2 = self.resnet.bn1(c0_img2)
       #  c0_img2 = self.resnet.relu(c0_img2)
       #  c1_img2 = self.resnet.maxpool(c0_img2)
       #  c1_img2 = self.resnet.layer1(c1_img2)
       #  # c1_img2 = self.drop(c1_img2)
       #  c2_img2 = self.resnet.layer2(c1_img2)
       #  # c2_img2 = self.drop(c2_img2)
       #  c3_img2 = self.resnet.layer3(c2_img2)
       #  c3_img2=self.resnet.maxpool(c3_img2)
       #  # c3_img2 = self.drop(c3_img2)
       #  c4_img2 = self.resnet.layer4(c3_img2)
       #  c4_img2=self.resnet.maxpool(c4_img2)
       #
       #  c0=self.dsres_lj1(c0)
       #  c1= self.dsres_lj2(c1)
       #  c2 = self.dsres_lj3(c2)
       #  c3 = self.dsres_lj4(c3)
       #  c4 = self.dsres_lj5(c4)
       #
       #  c0_img2 = self.dsres_lj1(c0_img2)
       #  c1_img2 = self.dsres_lj2(c1_img2)
       #  c2_img2 = self.dsres_lj3(c2_img2)
       #  c3_img2 = self.dsres_lj4(c3_img2)
       #  c4_img2 = self.dsres_lj5(c4_img2)
       #
       #
       #  # print(c0.shape)
       #  # print(c0_img2.shape)
       #  # lj=torch.cat((c0,c0_img2),dim=1)# 2 32 128 128
       #  # print(lj.shape)
       #  lj1=self.ds_lj1(torch.cat((c0,c0_img2),dim=1))#2 64 128 128
       #  lj2 = self.ds_lj2(torch.cat((c1, c1_img2), dim=1))#2 64 64 64
       #  lj3 = self.ds_lj3(torch.cat((c2, c2_img2), dim=1))#2 64 32 32
       #  lj4 = self.ds_lj4(torch.cat((c3, c3_img2), dim=1))#2 64 16 16
       #  lj5 = self.ds_lj5(torch.cat((c4, c4_img2), dim=1))#2 64 8 8
       #

       # print(lj1.shape)
       # print(lj2.shape)
       # print(lj3.shape)
       # print(lj4.shape)
       # print(lj5.shape)
       # forward backbone resnet

       x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
       x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
       #2 16 128 128
       #2 24 64 64
       #2 32 32 32
       #2 96 16 16
       #2 320 8 8
       # lj1 = torch.cat((x1_1, x2_1), dim=1)  # 2 64 128 128
       # lj2 = torch.cat((x1_2, x2_2), dim=1)  # 2 64 64 64
       # lj3 = torch.cat((x1_3, x2_3), dim=1)  # 2 64 32 32
       # lj4 = torch.cat((x1_4, x2_4), dim=1)  # 2 64 16 16
       # lj5 = torch.cat((x1_5, x2_5), dim=1)
       lj1 = self.ds_lj1(torch.cat((x1_1, x2_1), dim=1))  # 2 64 128 128
       lj2 = self.ds_lj2(torch.cat((x1_2, x2_2), dim=1))  # 2 64 64 64
       lj3 = self.ds_lj3(torch.cat((x1_3, x2_3), dim=1))  # 2 64 32 32
       lj4 = self.ds_lj4(torch.cat((x1_4, x2_4), dim=1))  # 2 64 16 16
       lj5 = self.ds_lj5(torch.cat((x1_5, x2_5), dim=1))  # 2 64 8 8

       # zy1 = self.ds_lj11(torch.abs(x1_1-x2_1)) # 2 64 128 128
       # zy2 = self.ds_lj22(torch.abs(x1_2-x2_2) ) # 2 64 64 64
       # zy3 = self.ds_lj33(torch.abs(x1_3-x2_3))  # 2 64 32 32
       zy4 = self.ds_lj44(torch.abs(x1_4-x2_4))  # 2 64 16 16
       zy5 = self.ds_lj55(torch.abs(x1_5-x2_5))  # 2 64 8 8

       #lj1 = self.ljds_lj1(torch.cat((x1_1, x2_1), dim=1))  # 2 64 128 128
       # lj2 = self.ljds_lj2(torch.cat((x1_2, x2_2), dim=1))  # 2 64 64 64
       # lj3 = self.ljds_lj3(torch.cat((x1_3, x2_3), dim=1))  # 2 64 32 32
       # lj4 = self.ljds_lj4(torch.cat((x1_4, x2_4), dim=1))  # 2 64 16 16
       # lj5 = self.ljds_lj5(torch.cat((x1_5, x2_5), dim=1))  # 2 64 8 8
       # x1_1=self.CBAM_ASPP1(x1_1)
       # x1_2 = self.CBAM_ASPP2(x1_2)
       # x1_3 = self.CBAM_ASPP3(x1_3)
       # x1_4 = self.CBAM_ASPP4(x1_4)
       # x1_5 = self.CBAM_ASPP5(x1_5)
       # x2_1 = self.CBAM_ASPP1(x2_1)
       # x2_2 = self.CBAM_ASPP2(x2_2)
       # x2_3 = self.CBAM_ASPP3(x2_3)
       # x2_4 = self.CBAM_ASPP4(x2_4)
       # x2_5= self.CBAM_ASPP5(x2_5)


       # print(x1_1.shape)#2 16 128 128
       # print(x1_2.shape)  #2 24 64 64
       # print(x1_3.shape)  #2 32 32 32
       # print(x1_4.shape)   # 2 96 16 16
       # print(x1_5.shape)    # 2 320 8 8
       # aaa = {'a', 'b', 'c', 'd'}
       # features, features11, features22 = [], [], []
       # for i in range(len(aaa)):
       #     if i == 0:
       #         features11.append(
       #             self.Cross_transformer_backbone_a00(x1_2, self.Cross_transformer_backbone_a0(x1_2, x2_2)))
       #         features22.append(
       #             self.Cross_transformer_backbone_b00(x2_2, self.Cross_transformer_backbone_b0(x2_2, x1_2)))
       #
       #     elif i == 1:
       #         features11.append(
       #             self.Cross_transformer_backbone_a11(x1_3, self.Cross_transformer_backbone_a1(x1_3, x2_3)))
       #         features22.append(
       #             self.Cross_transformer_backbone_b11(x2_3, self.Cross_transformer_backbone_a1(x2_3, x1_3)))
       #
       #     elif i == 2:
       #         features11.append(
       #             self.Cross_transformer_backbone_a22(x1_4, self.Cross_transformer_backbone_a2(x1_4, x2_4)))
       #         features22.append(
       #             self.Cross_transformer_backbone_b22(x2_4, self.Cross_transformer_backbone_a2(x2_4, x1_4)))
       #     elif i == 3:
       #
       #         features11.append(
       #             self.Cross_transformer_backbone_a33(x1_5, self.Cross_transformer_backbone_a3(x1_5, x2_5)))
       #         features22.append(
       #             self.Cross_transformer_backbone_b33(x2_5, self.Cross_transformer_backbone_a3(x2_5, x1_5)))

       # print(features11[0].shape)  # 2 24 64 64
       # print(features11[1].shape)  # 2 32 32 32
       # print(features11[2].shape)  # 2 96 16 16
       # print(features11[3].shape)  # 2 320 8 8
       # print(x1_2.shape)  # 2 24 64 64
       # print(x1_3.shape)  # 2 32 32 32
       # print(x1_4.shape)  # 2 96 16 16
       # print(x1_5.shape)  # 2 320 8 8


#更换nam的ResNeSt_Block
       # x1_2=self.ResNeSt_Block1(x1_2)
       # x1_3 = self.ResNeSt_Block2(x1_3)
       # x1_4 = self.ResNeSt_Block3(x1_4)
       # x1_5 = self.ResNeSt_Block4(x1_5)
       #
       # x2_2 = self.ResNeSt_Block1(x2_2)
       # x2_3 = self.ResNeSt_Block2(x2_3)
       # x2_4 = self.ResNeSt_Block3(x2_4)
       # x2_5 = self.ResNeSt_Block4(x2_5)
       #
       # x1_2 =self.rednestds_lyr1(x1_2)
       # x1_3= self.rednestds_lyr2(x1_3)
       # x1_4 = self.rednestds_lyr3(x1_4)
       # x1_5= self.rednestds_lyr4(x1_5)
       #
       # x2_2 = self.rednestds_lyr1(x2_2)
       # x2_3 = self.rednestds_lyr2(x2_3)
       # x2_4 = self.rednestds_lyr3(x2_4)
       # x2_5 = self.rednestds_lyr4(x2_5)
# 更换nam的ResNeSt_Block

#更换nam为m2snet

       # x1_dem_1=self.x1_dem_1(x1_1)
       # x2_dem_1 = self.x2_dem_1(x1_2)#2 64 64 64
       # x3_dem_1 = self.x3_dem_1(x1_3)#2 64 32 32
       # x4_dem_1 = self.x4_dem_1(x1_4)#2 64 16 16
       # x5_dem_1 = self.x5_dem_1(x1_5)#2 64 8 8
       #
       # x5_dem_1_up = F.upsample(x5_dem_1, size=x1_4.size()[2:], mode='bilinear')
       # x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
       # x4_dem_1_map1 = self.conv_3(x4_dem_1)
       # x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
       # x4_dem_1_map2 = self.conv_5(x4_dem_1)
       # x5_4 = self.x5_x4(
       #     abs(x5_dem_1_up - x4_dem_1) + abs(x5_dem_1_up_map1 - x4_dem_1_map1) + abs(x5_dem_1_up_map2 - x4_dem_1_map2))
       #
       # x4_dem_1_up = F.upsample(x4_dem_1, size=x1_3.size()[2:], mode='bilinear')
       # x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
       # x3_dem_1_map1 = self.conv_3(x3_dem_1)
       # x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
       # x3_dem_1_map2 = self.conv_5(x3_dem_1)
       # x4_3 = self.x4_x3(
       #     abs(x4_dem_1_up - x3_dem_1) + abs(x4_dem_1_up_map1 - x3_dem_1_map1) + abs(x4_dem_1_up_map2 - x3_dem_1_map2))
       #
       # x3_dem_1_up = F.upsample(x3_dem_1, size=x1_2.size()[2:], mode='bilinear')
       # x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
       # x2_dem_1_map1 = self.conv_3(x2_dem_1)
       # x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
       # x2_dem_1_map2 = self.conv_5(x2_dem_1)
       # x3_2 = self.x3_x2(
       #     abs(x3_dem_1_up - x2_dem_1) + abs(x3_dem_1_up_map1 - x2_dem_1_map1) + abs(x3_dem_1_up_map2 - x2_dem_1_map2))
       #
       # x2_dem_1_up = F.upsample(x2_dem_1, size=x1_1.size()[2:], mode='bilinear')
       # x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
       # x1_map1 = self.conv_3(x1_dem_1)
       # x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
       # x1_map2 = self.conv_5(x1_dem_1)
       # x2_1 = self.x2_x1(abs(x2_dem_1_up - x1_dem_1) + abs(x2_dem_1_up_map1 - x1_map1) + abs(x2_dem_1_up_map2 - x1_map2))
       #
       #
       # # print(x2_1.shape)
       # # y1_dem_1 = self.x1_dem_1(x2_1)
       # y1_dem_1=x2_1
       # y2_dem_1 = self.x2_dem_1(x2_2)  # 2 64 64 64
       # y3_dem_1 = self.x3_dem_1(x2_3)  # 2 64 32 32
       # y4_dem_1 = self.x4_dem_1(x2_4)  # 2 64 16 16
       # y5_dem_1 = self.x5_dem_1(x2_5)  # 2 64 8 8
       #
       # y5_dem_1_up = F.upsample(y5_dem_1, size=x2_4.size()[2:], mode='bilinear')
       # y5_dem_1_up_map1 = self.conv_3(y5_dem_1_up)
       # y4_dem_1_map1 = self.conv_3(y4_dem_1)
       # y5_dem_1_up_map2 = self.conv_5(y5_dem_1_up)
       # y4_dem_1_map2 = self.conv_5(y4_dem_1)
       # y5_4 = self.x5_x4(
       #     abs(y5_dem_1_up - y4_dem_1) + abs(y5_dem_1_up_map1 - y4_dem_1_map1) + abs(y5_dem_1_up_map2 - y4_dem_1_map2))
       #
       # y4_dem_1_up = F.upsample(y4_dem_1, size=x2_3.size()[2:], mode='bilinear')
       # y4_dem_1_up_map1 = self.conv_3(y4_dem_1_up)
       # y3_dem_1_map1 = self.conv_3(y3_dem_1)
       # y4_dem_1_up_map2 = self.conv_5(y4_dem_1_up)
       # y3_dem_1_map2 = self.conv_5(y3_dem_1)
       # y4_3 = self.x4_x3(
       #     abs(y4_dem_1_up - y3_dem_1) + abs(y4_dem_1_up_map1 - y3_dem_1_map1) + abs(y4_dem_1_up_map2 - y3_dem_1_map2))
       #
       # y3_dem_1_up = F.upsample(y3_dem_1, size=x2_2.size()[2:], mode='bilinear')
       # y3_dem_1_up_map1 = self.conv_3(y3_dem_1_up)
       # y2_dem_1_map1 = self.conv_3(y2_dem_1)
       # y3_dem_1_up_map2 = self.conv_5(y3_dem_1_up)
       # y2_dem_1_map2 = self.conv_5(y2_dem_1)
       # y3_2 = self.x3_x2(
       #     abs(y3_dem_1_up - y2_dem_1) + abs(y3_dem_1_up_map1 - y2_dem_1_map1) + abs(y3_dem_1_up_map2 - y2_dem_1_map2))
       #
       # y2_dem_1_up = F.upsample(y2_dem_1, size=x1_1.size()[2:], mode='bilinear')
       # y2_dem_1_up_map1 = self.conv_3(y2_dem_1_up)
       # y1_map1 = self.conv_3(y1_dem_1)
       # y2_dem_1_up_map2 = self.conv_5(y2_dem_1_up)
       # y1_map2 = self.conv_5(y1_dem_1)
       # y2_1 = self.x2_x1(
       #     abs(y2_dem_1_up - y1_dem_1) + abs(y2_dem_1_up_map1 - y1_map1) + abs(y2_dem_1_up_map2 - y1_map2))
       # x5_4 = self.resnet.maxpool(x5_4)
       # x4_3 = self.resnet.maxpool(x4_3)
       # x3_2 = self.resnet.maxpool(x3_2)
       # x2_1 = self.resnet.maxpool(x2_1)
       # y5_4 = self.resnet.maxpool(y5_4)
       # y4_3 = self.resnet.maxpool(y4_3)
       # y3_2 = self.resnet.maxpool(y3_2)
       # y2_1 = self.resnet.maxpool(y2_1)
# 更换nam为m2snet

#更换nam为深度可分离卷积(Depthwise Separable Convolution)

       # print(x1_1.shape)#2 16 128 128
       # print(x1_2.shape)  # 2 24 64 64
       # print(x1_3.shape)  # 2 32 32 32
       # print(x1_4.shape)  # 2 96 16 16
       # print(x1_5.shape)  # 2 320 8 8
       # print(lj1.shape)  # 2 16 128 128
       # print(lj2.shape)  # 2 24 64 64
       # print(lj3.shape)  # 2 32 32 32
       # print(lj4.shape)  # 2 96 16 16
       # print(lj5.shape)  # 2 320 8 8
#         x1_1 = self.depthsplit_conv1(x1_1)
#         x1_2 = self.depthsplit_conv2(x1_2)
#         x1_3 = self.depthsplit_conv3(x1_3)
#         x1_4= self.depthsplit_conv4(x1_4)
#         x1_5= self.depthsplit_conv5(x1_5)
#
#         x2_1 = self.depthsplit_conv1(x2_1)
#         x2_2 = self.depthsplit_conv2(x2_2)
#         x2_3 = self.depthsplit_conv3(x2_3)
#         x2_4 = self.depthsplit_conv4(x2_4)
#         x2_5 = self.depthsplit_conv5(x2_5)
#
#         x1_1 = self.resnet.maxpool(x1_1)
#         x1_2 = self.resnet.maxpool(x1_2)
#         x1_3= self.resnet.maxpool(x1_3)
#         x1_4 = self.resnet.maxpool(x1_4)
#
#         x2_1 = self.resnet.maxpool(x2_1)
#         x2_2 = self.resnet.maxpool(x2_2)
#         x2_3 = self.resnet.maxpool(x2_3)
#         x2_4 = self.resnet.maxpool(x2_4)
#
#         lj1= self.depthsplit_conv1(lj1)
#         lj2 = self.depthsplit_conv2(lj2)
#         lj3 = self.depthsplit_conv3(lj3)
#         lj4 = self.depthsplit_conv4(lj4)
#         lj5 = self.depthsplit_conv5(lj5)
#
#         lj1 = self.resnet.maxpool(lj1)
#         lj2 = self.resnet.maxpool(lj2)
#         lj3 = self.resnet.maxpool(lj3)
#         lj4 = self.resnet.maxpool(lj4)
#
# # 更换nam为深度可分离卷积(Depthwise Separable Convolution)
#
#         c4 = self.conv4_1(torch.cat([self.conv6_4_1(torch.cat([x1_4, x1_5], 1)), self.conv6_4_2(torch.cat([x1_4, x1_5], 1)),self.conv6_4_3(torch.cat([x1_4, x1_5], 1)), self.conv6_4_4(torch.cat([x1_4, x1_5], 1))], 1))
#         c4_1= F.interpolate(c4, scale_factor=(2, 2), mode='bilinear')
#         c3 = self.conv4_1(torch.cat([self.conv6_4_1(torch.cat([x1_3, c4_1], 1)), self.conv6_4_2(torch.cat([x1_3, c4_1], 1)),self.conv6_4_3(torch.cat([x1_3, c4_1], 1)), self.conv6_4_4(torch.cat([x1_3, c4_1], 1))], 1))
#         c3_1 = F.interpolate(c3, scale_factor=(2, 2), mode='bilinear')
#
#         c2 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([x1_2, c3_1], 1)), self.conv6_4_2(torch.cat([x1_2, c3_1], 1)),
#              self.conv6_4_3(torch.cat([x1_2, c3_1], 1)), self.conv6_4_4(torch.cat([x1_2, c3_1], 1))], 1))
#         c2_1 = F.interpolate(c2, scale_factor=(2, 2), mode='bilinear')
#
#         c1 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([x1_1, c2_1], 1)), self.conv6_4_2(torch.cat([x1_1, c2_1], 1)),
#              self.conv6_4_3(torch.cat([x1_1, c2_1], 1)), self.conv6_4_4(torch.cat([x1_1, c2_1], 1))], 1))
#         c1_1 = F.interpolate(c1, scale_factor=(2, 2), mode='bilinear')
#
#         cc4 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([x2_4, x2_5], 1)), self.conv6_4_2(torch.cat([x2_4, x2_5], 1)),
#              self.conv6_4_3(torch.cat([x2_4, x2_5], 1)), self.conv6_4_4(torch.cat([x2_4, x2_5], 1))], 1))
#         cc4_1 = F.interpolate(cc4, scale_factor=(2, 2), mode='bilinear')
#         cc3 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([x2_3, cc4_1], 1)), self.conv6_4_2(torch.cat([x2_3, cc4_1], 1)),
#              self.conv6_4_3(torch.cat([x2_3, cc4_1], 1)), self.conv6_4_4(torch.cat([x2_3, cc4_1], 1))], 1))
#         cc3_1 = F.interpolate(cc3, scale_factor=(2, 2), mode='bilinear')
#
#         cc2 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([x2_2, cc3_1], 1)), self.conv6_4_2(torch.cat([x2_2, cc3_1], 1)),
#              self.conv6_4_3(torch.cat([x2_2, cc3_1], 1)), self.conv6_4_4(torch.cat([x2_2, cc3_1], 1))], 1))
#         cc2_1 = F.interpolate(cc2, scale_factor=(2, 2), mode='bilinear')
#
#         cc1 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([x2_1, cc2_1], 1)), self.conv6_4_2(torch.cat([x2_1, cc2_1], 1)),
#              self.conv6_4_3(torch.cat([x2_1, cc2_1], 1)), self.conv6_4_4(torch.cat([x2_1, cc2_1], 1))], 1))
#         cc1_1 = F.interpolate(cc1, scale_factor=(2, 2), mode='bilinear')
#
#         ljc4 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([lj4,lj5], 1)), self.conv6_4_2(torch.cat([lj4,lj5], 1)),
#              self.conv6_4_3(torch.cat([lj4,lj5], 1)), self.conv6_4_4(torch.cat([lj4,lj5], 1))], 1))
#         ljc4_1 = F.interpolate(ljc4, scale_factor=(2, 2), mode='bilinear')
#         ljc3 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([lj3, ljc4_1], 1)), self.conv6_4_2(torch.cat([lj3, ljc4_1], 1)),
#              self.conv6_4_3(torch.cat([lj3, ljc4_1], 1)), self.conv6_4_4(torch.cat([lj3, ljc4_1], 1))], 1))
#         ljc3_1 = F.interpolate(ljc3, scale_factor=(2, 2), mode='bilinear')
#
#         ljc2 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([lj2,ljc3_1], 1)), self.conv6_4_2(torch.cat([lj2,ljc3_1], 1)),
#              self.conv6_4_3(torch.cat([lj2,ljc3_1], 1)), self.conv6_4_4(torch.cat([lj2,ljc3_1], 1))], 1))
#         ljc2_1 = F.interpolate(ljc2, scale_factor=(2, 2), mode='bilinear')
#
#         ljc1 = self.conv4_1(torch.cat(
#             [self.conv6_4_1(torch.cat([lj1, ljc2_1], 1)), self.conv6_4_2(torch.cat([lj1, ljc2_1], 1)),
#              self.conv6_4_3(torch.cat([lj1, ljc2_1], 1)), self.conv6_4_4(torch.cat([lj1, ljc2_1], 1))], 1))
#         ljc1_1 = F.interpolate(ljc1, scale_factor=(2, 2), mode='bilinear')
# # 更换nam为深度可分离卷积(Depthwise Separable Convolution)
#        x1_2=self.rednestds_lyr1(x1_2)
#        x2_2 = self.rednestds_lyr1(x2_2)
#        x1_3 = self.rednestds_lyr2(x1_3)
#        x2_3 = self.rednestds_lyr2(x2_3)
#        x1_4 = self.rednestds_lyr3(x1_4)
#        x2_4 = self.rednestds_lyr3(x2_4)
#        x1_5 = self.rednestds_lyr4(x1_5)
#        x2_5 = self.rednestds_lyr4(x2_5)

       # vis_feat(x2_2, 4, 5, '/home/slr/mount/lunwen/A2Netjieguotu/SYSUhot', 'modelyuan.png')
       x1_2=self.LCC2(x1_2)
       x1_3 = self.LCC3(x1_3)
       # x1_4 = self.LCC4(x1_4)
       # x1_5 = self.LCC5(x1_5)
       # x1_4=self.LSR4(x1_4)
       # x1_5 = self.LSR5(x1_5)
       # x1_4 = F.interpolate(x1_4, size=(16, 16), mode='bilinear', align_corners=False)
       # x1_5 = F.interpolate(x1_5, size=(8, 8), mode='bilinear', align_corners=False)
       x2_2 = self.LCC2(x2_2)
       x2_3 = self.LCC3(x2_3)

       zy2 = torch.abs(x1_2 - x2_2)
       zy3 = torch.abs(x1_3 - x2_3)
       # x2_4 = self.LSR4(x2_4)
       # x2_5 = self.LSR5(x2_5)
       # x2_4 = F.interpolate(x2_4, size=(16, 16), mode='bilinear', align_corners=False)
       # x2_5 = F.interpolate(x2_5, size=(8, 8), mode='bilinear', align_corners=False)
       # x2_4 = self.LCC4(x2_4)
       # x2_5 = self.LCC5(x2_5)


       # x1_2, x1_3, x1_4, x1_5 = self.swa(x1_2, x1_3, x1_4, x1_5)
       # x2_2, x2_3, x2_4, x2_5 = self.swa(x2_2, x2_3, x2_4, x2_5)  # neiborhood aggre
       # vis_feat(x2_2, 5, 5, 'D:\slr\A2Net\A2Net-main\models/results_shuWHU256_iter_39998_lr_0.0005', 'x1_2.png')

       # print(x1_2.shape)  # 2 64 64 64
       # print(x1_3.shape)  # 2 64 32 32
       # print(x1_4.shape)  # 2 64 16 16
       # print(x1_5.shape)  # 2 64 8 8
       ## temporal fusion
       # c2, c3, c4, c5 = self.tfm(c1, c2, c3, c4, cc1, cc2, cc3, cc4)
       # c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)

#        x1_2, x1_3, x1_4, x1_5 = self.swa(x1_2, x1_3, x1_4, x1_5)
#        x2_2, x2_3, x2_4, x2_5 = self.swa(x2_2, x2_3, x2_4, x2_5)  # neiborhood aggre


       # print(x1_2.shape)  # 2 64 64 64
       # print(x1_3.shape)  # 2 64 32 32
       # print(x1_4.shape)  # 2 64 16 16
       # print(x1_5.shape)  # 2 64 8 8
       ## temporal fusion
       #c2, c3, c4, c5 = self.tfm(c1, c2, c3, c4, cc1, cc2, cc3, cc4)
       #c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)

       # 并行池化
       c5, c4 = self.inctfm(zy5,zy4)
       db1=self.enhan1(zy3)
       sb1=self.enhan1(c4)
       db2=self.adjdb(zy2)
       sb2=self.adjsb(c5)
       # c2=self.haole(c2)
       # c3 = self.haole(c3)
       # c4 = self.haole(c4)
       # c5= self.haole(c5)

       # c2 = self.s2att(c2)
       # c3 = self.s2att(c3)
       # c4 = self.s2att(c4)
       # c5 = self.s2att(c5)

       # c2 = self.coord(c2)
       # c3 = self.coord(c3)
       # c4 = self.coord(c4)
       # c5 = self.coord(c5)

       #c2, c3, c4, c5 = self.inctfminvolution(c2,c3,c4,c5)

       # c2 = self.DoubleAttent(c2)
       # c3 = self.DoubleAttent(c3)
       # c4 = self.DoubleAttent(c4)
       # c5 = self.DoubleAttent(c5)

       #c2 = self.shuffleattention(c2)
       # c3 = self.shuffleattention(c3)
       # c4 = self.shuffleattention(c4)
       # c5 = self.shuffleattention(c5)
       #cm2, cm3, cm4, cm5 = self.dynamictfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
       #c2, c3, c4, c5 = self.tfm(x2_1,x3_2,x4_3,x5_4, y2_1 ,y3_2, y4_3, y5_4,  )
## 更换nam为深度可分离卷积(Depthwise Separable Convolution)
       # print(ljc1_1.shape)#2 64 128 128
       # print(ljc2_1.shape)  # 2 64 64 64
       # print(ljc3_1.shape)  # 2 64 32 32
       # print(ljc4_1.shape)  # 2 64 16 16

       #c2, c3, c4, c5 = self.tfm(c1, c2, c3, c4, cc1, cc2, cc3, cc4)
       #ljc2, ljc3, ljc4, ljc5 = self.ljtfm(ljc1_1,ljc2_1,ljc3_1,ljc4_1)
       # c1, c2, c3, c4 = self.ljtfm(c1, c2, c3, c4)
       # cc1, cc2, cc3, cc4 = self.ljtfm(cc1, cc2, cc3, cc4)
#lj直流全变
       # ljc2 = self.resnet.maxpool(ljc2)
       # ljc3 = self.resnet.maxpool(ljc3)
       # ljc4= self.resnet.maxpool(ljc4)
       # ljc5 = self.resnet.maxpool(ljc5)
# lj直流全变
       # print(c2.shape) #2 64 64 64
       # print(c3.shape) #2 64 32 32
       # print(c4.shape)#2 64 16 16
       # print(c5.shape)#2 64 8 8
       # print(ljc2.shape) #2 64 64 64
       # print(ljc3.shape) #2 64 32 32
       # print(ljc4.shape)#2 64 16 16
       # print(ljc5.shape)#2 64 8 8
       # fpn

       # print(lj3.shape)


       # c2 = self.tripbletatten(c2)
       # c3 = self.tripbletatten(c3)
       # c4 = self.tripbletatten(c4)
       # c5 = self.tripbletatten(c5)
       # c2,c3,c4,c5=self.seweight(c2,c3,c4,c5)
       # lj2,lj3,lj4,lj5 = self.seweight(lj2,lj3,lj4,lj5)
       # zyw2, zyw3, zyw4, zyw5 = self.seweight(zy2, zy3, zy4, zy5)
       # lj2=lj2*zyw2
       # c2=c2*zyw2
       # lj3=lj3*zyw3
       # c3=c3*zyw3
       t21, t22, t23, t24, t25, t26 = self.tripbletripletsange(db1,sb2)
       t31, t32, t33, t34, t35, t36 = self.tripbletripletsange(db2,sb1)
       # t41, t42, t43, t44, t45, t46 = self.tripbletripletsange(c4, lj4)
       # t51, t52, t53, t54, t55, t56 = self.tripbletripletsange(c5, lj5)

       cross_21, curg_21, curl_21 = self.cross2(t21, t24)
       cross_22, curg_22, curl_22 = self.cross2(t22, t25)
       cross_23, curg_23, curl_23 = self.cross2(t23, t26)
       cross_2 = cross_21 + cross_22 + cross_23
       curg_2 = curg_21 + curg_22 + curg_23
       curl_2 = curl_21 + curl_22 + curl_23

       cross_31, curg_31, curl_31 = self.cross3(t31, t34)
       cross_32, curg_32, curl_32 = self.cross3(t32, t35)
       cross_33, curg_33, curl_33 = self.cross3(t33, t36)
       cross_3 = cross_31 + cross_32 + cross_33
       curg_3 = curg_31 + curg_32 + curg_33
       curl_3 = curl_31 + curl_32 + curl_33

       # cross_41, curg_41, curl_41 = self.cross4(t41, t44)
       # cross_42, curg_42, curl_42 = self.cross4(t42, t45)
       # cross_43, curg_43, curl_43 = self.cross4(t43, t46)
       # cross_4 = cross_41 + cross_42 + cross_43
       # curg_4 = curg_41 + curg_42 + curg_43
       # curl_4 = curl_41 + curl_42 + curl_43
       #
       # cross_51, curg_51, curl_51 = self.cross5(t51, t54)
       # cross_52, curg_52, curl_52 = self.cross5(t52, t55)
       # cross_53, curg_53, curl_53 = self.cross5(t53, t56)
       # cross_5 = cross_51 + cross_52 + cross_53
       # curg_5 = curg_51 + curg_52 + curg_53
       # curl_5 = curl_51 + curl_52 + curl_53


       # vis_feat(cross_22, 8, 8, '/home/slr/mount/lunwen/A2Netjieguotu/SYSUhot', 'c2aftertriple.png')

# #tripleattanion三分支和triple分支cross
#         t21,t22,t23 = self.tripbletriplet(c2)
#         t31,t32,t33 = self.tripbletriplet(c3)
#         t41,t42,t43 = self.tripbletriplet(c4)
#         t51,t52,t53 = self.tripbletriplet(c5)
#         zz2 = c2
#         zz3 = c3
#         zz4 = c4
#         zz5 = c5
#         c2 = self.ds_lyr5(torch.cat([z2, c2], dim=1))  # 2 6 64 64
#         c3 = self.ds_lyr5(torch.cat([z3, c3], dim=1))
#         c4 = self.ds_lyr5(torch.cat([z4, c4], dim=1))
#         c5 = self.ds_lyr5(torch.cat([z5, c5], dim=1))
#
#         # cross_2, curg_2, curl_2 = self.cross2(lj2, zz2)  # 2 64 64 64
#         # cross_3, curg_3, curl_3 = self.cross3(lj3, zz3)  # 2 64 32 32
#         # cross_4, curg_4, curl_4 = self.cross4(lj4, zz4)  # 2 64 16 16
#         # cross_5, curg_5, curl_5 = self.cross5(lj5, zz5)  # 2 64 8 8
#
#
#
#
#         cross_21, curg_21, curl_21 = self.cross2(lj2,t21)
#         cross_22, curg_22, curl_22 =self.cross2(lj2,t22)
#         cross_23, curg_23, curl_23 =self.cross2(lj2,t23)
#         cross_2=cross_21+cross_22+cross_23
#         curg_2=curg_21+curg_22+curg_23
#         curl_2=curl_21+curl_22+curl_23#2 64 64 64
#
#         cross_31, curg_31, curl_31 = self.cross3(lj3, t31)
#         cross_32, curg_32, curl_32 = self.cross3(lj3, t32)
#         cross_33, curg_33, curl_33 = self.cross3(lj3, t33)
#         cross_3 = cross_31 + cross_32 + cross_33
#         curg_3 = curg_31 + curg_32 + curg_33
#         curl_3 = curl_31 + curl_32 + curl_33  # 2 64 64 64
#
#         cross_41, curg_41, curl_41 = self.cross4(lj4, t41)
#         cross_42, curg_42, curl_42 = self.cross4(lj4, t42)
#         cross_43, curg_43, curl_43 = self.cross4(lj4, t43)
#         cross_4 = cross_41 + cross_42 + cross_43
#         curg_4 = curg_41 + curg_42 + curg_43
#         curl_4 = curl_41 + curl_42 + curl_43  # 2 64 64 64
#
#         cross_51, curg_51, curl_51 = self.cross5(lj5, t51)
#         cross_52, curg_52, curl_52 = self.cross5(lj5,t52)
#         cross_53, curg_53, curl_53 = self.cross5(lj5, t53)
#         cross_5 = cross_51 + cross_52 + cross_53
#         curg_5 = curg_51 + curg_52 + curg_53
#         curl_5 = curl_51 + curl_52 + curl_53  # 2 64 64 64

       # y2 = self.ds_lyr5(torch.cat([cross_2, c2], dim=1))
       # y3 = self.ds_lyr5(torch.cat([cross_3, c3], dim=1))
       # y4 = self.ds_lyr5(torch.cat([cross_4, c4], dim=1))
       # y5 = self.ds_lyr5(torch.cat([cross_5, c5], dim=1))
#TFIM融合
       # d5=self.TFIM(curg_2,curl_2)
       # d4 = self.TFIM(curg_3, curl_3)
       # d3 = self.TFIM(curg_4, curl_4)
       # d2 = self.TFIM(curg_5, curl_5)
# #TFIM
#         # FHD融合
#         agg2 = cross_2+c2
#         curg_2 = self.la_layers(agg2)
#         curl_2 = self.ga_layers(agg2)
#         d2 = curg_2+curl_2  # 2 64 64 64
#
#         agg3 = cross_3+c3
#         curg_3 = self.la_layers(agg3)
#         curl_3 = self.ga_layers(agg3)
#         d3 = curg_3+curl_3  # 2 64 32 32
#
#         agg4 = cross_4+c4  # 2 64 16 16
#         curg_4 = self.la_layers(agg4)
#         curl_4 = self.ga_layers(agg4)
#         d4 = curg_4+curl_4  # 2 64 16 16
#
#         agg5 = cross_5+c5
#         curg_5 = self.la_layers(agg5)
#         curl_5 = self.ga_layers(agg5)
#         d5 = curg_5+curl_5  # 2 64 8 8
#
#         d2 = torch.sigmoid(d2)
#         diff2 = 2 * cross_2 * d2 + 2 * c2 * (1 - d2)
#         d2 = F.interpolate(d2, scale_factor=(4, 4), mode='bilinear')
#
#         d3 = torch.sigmoid(d3)
#         diff3 = 2 * cross_3 * d3 + 2 * c3 * (1 - d3)
#         d3 = F.interpolate(d3, scale_factor=(8, 8), mode='bilinear')
#
#         d4 = torch.sigmoid(d4)
#         diff4 = 2 * cross_4 * d4 + 2 * c4 * (1 - d4)
#         d4 = F.interpolate(d4, scale_factor=(16, 16), mode='bilinear')
#
#         d5 = torch.sigmoid(d5)
#         diff5 = 2 * cross_5 * d5 + 2 * c5 * (1 - d5)
#         d5 = F.interpolate(d5, scale_factor=(32, 32), mode='bilinear')
#         # FHD融合结束
# # #FHD融合
#         agg2=curg_2+curl_2
#         curg_2 = self.la_layers(agg2)
#         curl_2 = self.ga_layers(agg2)
#         d2 = curl_2+curg_2#2 64 64 64
#
#         agg3 = curg_3 + curl_3
#         curg_3 = self.la_layers(agg3)
#         curl_3 = self.ga_layers(agg3)
#         d3 = curl_3 + curg_3  # 2 64 32 32
#
#         agg4 = curg_4 + curl_4  # 2 64 16 16
#         curg_4 = self.la_layers(agg4)
#         curl_4 = self.ga_layers(agg4)
#         d4 = curl_4 + curg_4  # 2 64 16 16
#
#         agg5 = curg_5 + curl_5
#         curg_5= self.la_layers(agg5)
#         curl_5 = self.ga_layers(agg5)
#         d5 = curl_5 + curg_5  # 2 64 8 8
#
#         d2 = torch.sigmoid(d2)
#         diff2 = 2 * curg_2 * d2 + 2 * curl_2 * (1 - d2)
#         d2 = F.interpolate(d2, scale_factor=(4, 4), mode='bilinear')
#
#         d3 = torch.sigmoid(d3)
#         diff3 = 2 * curg_3 * d3 + 2 * curl_3 * (1 - d3)
#         d3 = F.interpolate(d3, scale_factor=(8, 8), mode='bilinear')
#
#         d4 = torch.sigmoid(d4)
#         diff4 = 2 * curg_4 * d4 + 2 * curl_4 * (1 - d4)
#         d4 = F.interpolate(d4, scale_factor=(16, 16), mode='bilinear')
#
#         d5 = torch.sigmoid(d5)
#         diff5 = 2 * curg_5 * d5 + 2 * curl_5 * (1 - d5)
#         d5 = F.interpolate(d5, scale_factor=(32, 32), mode='bilinear')
# #FHD融合结束
#temp 源代码的decoder
       p2, p3, mask_p2, mask_p3= self.decoder(cross_2, cross_3)  # SAM
       #p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)
       # d4 = self.decode4(cross_5)
       # d3 = self.decode4(d4 + cross_4)
       # d2 = self.decode4(d3 + cross_3)
       # d1 = self.decode4(d2 + cross_2)
       # f1 = self.deconv_out1(d1)
       # f2 = self.conv_out(f1)
       # mask_p2= self.deconv_out2(f2)
       #
       # f21 = self.deconv_out1(d2)
       # f22 = self.conv_out(f21)
       # f23 = self.deconv_out2(f22)
       # mask_p3 = self.deconv_out3(f23)
       #
       # f31 = self.deconv_out1(d3)
       # f32 = self.conv_out(f31)
       # f33 = self.deconv_out2(f32)
       # f33 = self.deconv_out3(f33)
       # mask_p4 = self.deconv_out3(f33)
       #
       # f41 = self.deconv_out1(d4)
       # f42 = self.conv_out(f41)
       # f43 = self.deconv_out2(f42)
       # f43 = self.deconv_out3(f43)
       # f43 = self.deconv_out3(f43)
       # mask_p5 = self.deconv_out3(f43)

#temp
       #pm2, pm3, pm4, pm5, maskm_p2, maskm_p3, maskm_p4, maskm_p5 = self.decoder(cm2,cm3,cm4,cm5)
       #xx=torch.cat([mask_p2,maskm_p2],dim=1)
       # mask_p2 = self.ds_lyr6(torch.cat([mask_p2, maskm_p2], dim=1))
       # mask_p3 = self.ds_lyr6(torch.cat([mask_p3, maskm_p3], dim=1))
       # mask_p4 = self.ds_lyr6(torch.cat([mask_p4, maskm_p4], dim=1))
       # mask_p5 = self.ds_lyr6(torch.cat([mask_p5, maskm_p5], dim=1))
       #p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(diff2,diff3,diff4,diff5)
       # print(c2.shape) 2 64 64 64
       # print(c3.shape) 2 64 32 32
       # print(mask_p2.shape) #2 1 64 64
       # print(mask_p3.shape)  # 2 1 32  32
       # change map
#临时去除decoder假的
       # cross_5 = self.ds_lyr4(cross_5)
       # cross_4= self.ds_lyr4(cross_4)
       # cross_3= self.ds_lyr4(cross_3)
       # cross_2= self.ds_lyr4(cross_2)
       # cross_2 = F.interpolate(cross_2, scale_factor=(4, 4), mode='bilinear')
       # cross_2 = torch.sigmoid(cross_2)
       # cross_3 = F.interpolate(cross_3, scale_factor=(8, 8), mode='bilinear')
       # cross_3= torch.sigmoid(cross_3)
       # cross_4 = F.interpolate(cross_4, scale_factor=(16, 16), mode='bilinear')
       # cross_4= torch.sigmoid(cross_4)
       # cross_5 = F.interpolate(cross_5, scale_factor=(32, 32), mode='bilinear')
       # cross_5= torch.sigmoid(cross_5)
#临时去除decoder假的
       mask_p2 = F.interpolate(mask_p2, scale_factor=(8, 8), mode='bilinear')
       mask_p2 = torch.sigmoid(mask_p2)
       #
       #
       mask_p3 = F.interpolate(mask_p3, scale_factor=(16, 16), mode='bilinear')
       mask_p3 = torch.sigmoid(mask_p3)
       #
       # mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
       # mask_p4 = torch.sigmoid(mask_p4)
       # #
       # #
       # mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
       # mask_p5 = torch.sigmoid(mask_p5)

       # maskm_p2 = F.interpolate(maskm_p2, scale_factor=(4, 4), mode='bilinear')
       # maskm_p2 = torch.sigmoid(maskm_p2)
       #
       # maskm_p3 = F.interpolate(maskm_p3, scale_factor=(8, 8), mode='bilinear')
       # maskm_p3 = torch.sigmoid(maskm_p3)
       #
       # maskm_p4 = F.interpolate(maskm_p4, scale_factor=(16, 16), mode='bilinear')
       # maskm_p4 = torch.sigmoid(maskm_p4)
       #
       # maskm_p5 = F.interpolate(maskm_p5, scale_factor=(32, 32), mode='bilinear')
       # maskm_p5 = torch.sigmoid(maskm_p5)

       # print(mask_p2.shape)  # 2 1 256 256
       # print(mask_p3.shape)#2 1 256 256
       # x = torch.cat([d5, d4, d3, d2], dim=1)
       # x_ca = self.ca(x)
       # # x = x * x_ca
       # # x = self.conv_dr(x)
       # # x=self.ds_lyr4(x)
       # # x = torch.sigmoid(x)
       # x = torch.cat([cross_5,cross_4,cross_3,cross_2], dim=1)
       # x_ca = self.ca(x)
       # x = x * x_ca
       # x = self.conv_dr(x)
       # x = self.ds_lyr4(x)
       # x = torch.sigmoid(x)
#FHDtemp
       # x = torch.cat([maskm_p5,maskm_p4,maskm_p3,maskm_p2], dim=1)
       # x=self.ds_lyr41(x)
       # x = torch.sigmoid(x)
       #d4=self.ds_lyr4(d4)
       # d3=self.ds_lyr4(d3)
       # d2=self.ds_lyr4(d2)
       # x = self.conv_fusion(torch.cat((d2,d3,d4,d5), dim=1))
       # x= torch.sigmoid(x)
#FHD
#第一二层的监督
       # ds2 = self.ds_lyr2(torch.abs(x1_1 - x2_1))  # 2 1 256 256
       # # print(torch.abs(x1_2 - x2_2).shape)
       # ds3 = self.ds_lyr3(torch.abs(x1_2 - x2_2))

       #

       # ds3 = self.ds_lyr3(torch.abs(x1_2 - x2_2))
#第一二层的监督  #
       # print(ds3.shape)
       # x1_1_1 = F.interpolate(x1_1, scale_factor=(4, 4), mode='bilinear')
       # ds2=torch.abs(d5-d4)
       # ds2 = F.interpolate(ds2, scale_factor=(1, 1), mode='bilinear')
       # ds2 = torch.sigmoid(ds2)
       # ds3=torch.abs(d3-d2)
       # ds3 = F.interpolate(ds3, scale_factor=(1, 1), mode='bilinear')
       # ds3 = torch.sigmoid(ds3)
       # cross_5 = self.ds_lyr4(cross_5)
       # cross_4= self.ds_lyr4(cross_4)
       # cross_3= self.ds_lyr4(cross_3)
       # cross_2= self.ds_lyr4(cross_2)
       # cross_2 = torch.sigmoid(cross_2)
       # cross_3 = torch.sigmoid(cross_3)
       # cross_4 = torch.sigmoid(cross_4)
       # cross_5 = torch.sigmoid(cross_5)
       return mask_p2, mask_p3
       #return cross_5,cross_4,cross_3,cross_2
import sys

sys.path.append("../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import os
import math
from thop import profile

try:
    import TwinMixing.config as cfg
    from TwinMixing.utils import *
    from TwinMixing.modules import *
except:
    import config as cfg
    from utils import *
    from modules import *

import sys



DILATION_RATE = None
GROUPS = None

class ShuffleUnitBR(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, dilation=1, groups=2):
        super(ShuffleUnitBR, self).__init__()
        padding = int((kernel_size - 1) / 2)
        mid = int(nout // 2)
        
        groups = math.gcd(math.gcd(nin, mid), math.gcd(mid,nout))
        if GROUPS is not None:
            if groups % GROUPS == 0:
                groups = GROUPS

        self.conv1 = nn.Conv2d(nin, mid, 1, 1, 0, 1, groups = groups, bias=False)
        self.conv3 = nn.Conv2d(mid, nout, 1, 1, 0, 1, groups = groups, bias=False)
        self.groups = groups
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.PReLU(mid)
        if stride == 2:
            self.conv2 = nn.Conv2d(mid, mid, kernel_size, stride, padding, dilation, groups=mid, bias=False)
            self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv2 = nn.Conv2d(mid, mid, kernel_size, stride, padding, dilation, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.bn3 = nn.BatchNorm2d(nout)
        if stride == 2:
            self.act2 = nn.PReLU(nout + nin)
        else:
            self.act2 = nn.PReLU(nout)
    
    def forward(self, x):
        
        if hasattr(self, 'avg_pool'):
            out1 = self.avg_pool(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if hasattr(self, 'avg_pool'):
            out = torch.cat([out, out1], dim=1)
        elif x.shape == out.shape:
            out = torch.add(out, x)
        
        out = self.act2(out)
        return out

class ShuffleUnit(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, dilation=1):
        super(ShuffleUnit, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        mid = int(nout // 2)
        
        groups = math.gcd(math.gcd(nin, mid), math.gcd(mid,nout))
        if GROUPS is not None:
            if groups % GROUPS == 0:
                groups = GROUPS
        self.conv1 = nn.Conv2d(nin, mid, 1, 1, 0, 1, groups = groups, bias=False)
        self.conv3 = nn.Conv2d(mid, nout, 1, 1, 0, 1, groups = groups, bias=False)
        self.groups = groups
        
        if stride == 2:
            self.conv2 = nn.Conv2d(mid, mid, kernel_size, stride, padding, dilation, groups=mid, bias=False)
            self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv2 = nn.Conv2d(mid, mid, kernel_size, stride, padding, dilation, groups=mid, bias=False)
    
    def forward(self, x):
        
        if hasattr(self, 'avg_pool'):
            out1 = self.avg_pool(x)
        
        out = self.conv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if hasattr(self, 'avg_pool'):
            out = torch.cat([out, out1], dim=1)
        elif x.shape == out.shape:
            out = torch.add(out, x)
        
        return out

class StrideEPM(nn.Module):
    def __init__(self, nIn, nOut, area=1):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = ShuffleUnit(nIn, n, 3, 2, 1)
        
        if DILATION_RATE is None:
            self.d1 = ShuffleUnit(n + nIn, n1, 3, 1, 1)
            self.d2 = ShuffleUnit(n + nIn, n, 3, 1, 2)
            self.d4 = ShuffleUnit(n + nIn, n, 3, 1, 4)
            self.d8 = ShuffleUnit(n + nIn, n, 3, 1, 8)
            self.d16 = ShuffleUnit(n + nIn, n, 3, 1, 16)
        else:
            d = DILATION_RATE
            self.d1 = ShuffleUnit(n + nIn, n1, 3, 1, d)
            self.d2 = ShuffleUnit(n + nIn, n, 3, 1, d)
            self.d4 = ShuffleUnit(n + nIn, n, 3, 1, d)
            self.d8 = ShuffleUnit(n + nIn, n, 3, 1, d)
            self.d16 = ShuffleUnit(n + nIn, n, 3, 1, d)
        
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        output = self.bn(combine)
        output = self.act(output)
        return output

class EPM(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True, area=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        self.c1 = ShuffleUnit(nIn, n, 1, 1, 1)
        
        if DILATION_RATE is None:
            self.d1 = ShuffleUnit(n, n1, 3, 1, 1) # dilation rate of 2^0
            self.d2 = ShuffleUnit(n, n, 3, 1, 2) # dilation rate of 2^1
            self.d4 = ShuffleUnit(n, n, 3, 1, 4) # dilation rate of 2^2
            self.d8 = ShuffleUnit(n, n, 3, 1, 8) # dilation rate of 2^3
            self.d16 = ShuffleUnit(n, n, 3, 1, 16) # dilation rate of 2^4
        else:
            d = DILATION_RATE
            self.d1 = ShuffleUnit(n, n1, 3, 1, d) # dilation rate of 2^0
            self.d2 = ShuffleUnit(n, n, 3, 1, d) # dilation rate of 2^1
            self.d4 = ShuffleUnit(n, n, 3, 1, d) # dilation rate of 2^2
            self.d8 = ShuffleUnit(n, n, 3, 1, d) # dilation rate of 2^3
            self.d16 = ShuffleUnit(n, n, 3, 1, d) # dilation rate of 2^4
        
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = torch.add(input, combine)
        output = self.bn(combine)
        return output

class DualBranchUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False,kernel_size = 3):
        super(DualBranchUpConvBlock, self).__init__()
        self.last=last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        
        self.cg_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Upsample(size=None, scale_factor=2, mode="bilinear"),
            nn.PReLU(out_channels),
        )
        if not last:
            self.conv1 = CBR(out_channels+sub_dim,out_channels,kernel_size, 1, 1)
    def forward(self, x, ori_img=None):
        # fine detailed branch
        fd = self.up_conv(x)
        if not self.last:
            fd = torch.cat([fd, ori_img], dim=1)
            fd = self.conv1(fd)
        # coarse grained branch
        cg = self.cg_branch(x)
        x = fd + cg
        # x = cg
        return x

class Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, config):
        super().__init__()
        chanel_img = cfg.channel_img
        


        planes = cfg.sc_ch_dict[config]["planes"]
        self.level1 = ShuffleUnitBR(chanel_img, planes, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = ShuffleUnitBR(planes + chanel_img * 2, planes * 2, 3)
        # self.b1 = CBR(planes + chanel_img, planes * 2)
        self.level2_0 = StrideEPM(planes * 2, planes * 2 * 2, area = 8)

        self.level2 = nn.ModuleList()
        for i in range(0, cfg.sc_ch_dict[config]["p"]):
            self.level2.append(EPM(planes * 2 * 2 , planes * 2 * 2, area = 8))
        self.b2 = ShuffleUnitBR(planes * 2 * 2 * 2 + chanel_img, planes * 2 * 2 * 2 + chanel_img, 3, 1)

        self.level3_0 = StrideEPM(planes * 2 * 2 * 2 + chanel_img, planes * 2 * 2 * 2, area = 4)
        self.level3 = nn.ModuleList()
        for i in range(0, cfg.sc_ch_dict[config]["q"]):
            self.level3.append(EPM(planes * 2 * 2 * 2, planes * 2 * 2 * 2, area = 4))
        self.b3 = ShuffleUnitBR(planes * 2 * 2 * 2 * 2, planes * 2 * 2, 3, 1)
        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        # print(output0_cat.shape)
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        # print(output1_cat.shape)
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        # print(output2.shape)
        output2_cat=torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)
        
        return out_encoder,output0_cat,output1_cat,output2_cat,inp1,inp2

class TwinMixing(nn.Module):
    '''
    This class defines the ShuffleESPNet network
    '''

    def __init__(self, args=None, student=False):

        super().__init__()
        print("==============================")
        print("TwinMixing")
        print("==============================")
        if not student:
            model_cfg = cfg.sc_ch_dict[args.teacher_type]
            self.encoder = Encoder(args.teacher_type)
        else:
            model_cfg = cfg.sc_ch_dict[args.student_type]
            self.encoder = Encoder(args.student_type)

        print(model_cfg)

        
        planes = model_cfg["planes"]

        self.caam = CAAM(feat_in=planes * 2 * 2, num_classes=planes * 2 * 2,bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ShuffleUnitBR(planes * 2 * 2, planes * 2, 3, 1)

        self.up_1_da = DualBranchUpConvBlock(planes * 2, planes) #out: Hx4, Wx4
        self.up_2_da = DualBranchUpConvBlock(planes, 8) #out: Hx2, Wx2
        self.out_da = DualBranchUpConvBlock(8, 2,last=True)  

        self.up_1_ll = DualBranchUpConvBlock(planes * 2, planes) #out: Hx4, Wx4
        self.up_2_ll = DualBranchUpConvBlock(planes, 8) #out: Hx2, Wx2
        self.out_ll = DualBranchUpConvBlock(8, 2,last=True)
        


    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        out_encoder,output0,output1,output2,inp1,inp2=self.encoder(input)

        out_caam=self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)

        out_da=self.up_1_da(out_caam,inp2)
        out_da=self.up_2_da(out_da,inp1)
        out_da_logit=self.out_da(out_da)

        out_ll=self.up_1_ll(out_caam,inp2)
        out_ll=self.up_2_ll(out_ll,inp1)
        out_ll_logit=self.out_ll(out_ll)
            


        return out_da_logit, out_ll_logit,output0,output1,out_encoder

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
import time
def time_c():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def benchmark_fps(
    model,
    example,
    batch_size: int,
    warmup: int = 50,
    iters: int = 500,
    repeats: int = 10,
    measure_per_iter: bool = True,
):
    model.eval()

    # --- Warmup ---
    for _ in range(warmup):
        _ = model(example)

    fps_batches = []
    fps_images = []

    for _ in range(repeats):
        t0 = time.perf_counter()

        for _ in range(iters):
            _ = model(example)

        dt = time.perf_counter() - t0

        fps_batch = iters / dt              # batches/sec
        fps_img = fps_batch * batch_size    # images/sec
        fps_batches.append(fps_batch)
        fps_images.append(fps_img)

    fps_batches = np.array(fps_batches, dtype=np.float64)
    fps_images = np.array(fps_images, dtype=np.float64)

    print(f"[Repeat-level] Batch FPS: mean={fps_batches.mean():.2f}, std={fps_batches.std(ddof=1):.2f} (n={repeats})")
    print(f"[Repeat-level] Image FPS: mean={fps_images.mean():.2f}, std={fps_images.std(ddof=1):.2f} (n={repeats})")

    if measure_per_iter:
        times_ms = []

        for _ in range(iters):
            t0 = time.perf_counter()

            _ = model(example)

            dt = 1/(time.perf_counter() - t0)
            times_ms.append(dt)

        times_ms = np.array(times_ms, dtype=np.float64)
        
        per_image_ms = times_ms * batch_size

        print(f"[Iter-level] Batch latency (fps): mean={times_ms.mean():.3f}, std={times_ms.std(ddof=1):.3f} (n={iters})")
        print(f"[Iter-level] Img latency (fps):   mean={per_image_ms.mean():.3f}, std={per_image_ms.std(ddof=1):.3f} (n={iters})")

        fps_img_from_latency = 1 / per_image_ms.mean()
        print(f"[Iter-level] Approx Image FPS from mean latency: {fps_img_from_latency:.2f}")

    return fps_batches, fps_images
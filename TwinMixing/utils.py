import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import os
import math

def second_largest_common_divisor(a: int, b: int) -> int:
    """Trả về ước số chung lớn thứ nhì của hai số a và b."""
    gcd_ab = math.gcd(a, b)
    # Tìm tất cả ước số của gcd_ab (tức là các ước chung của a và b)
    divisors = [i for i in range(1, gcd_ab + 1) if gcd_ab % i == 0]
    if len(divisors) < 2:
        return None  # Không có ước số lớn thứ nhì
    return divisors[-2]  # Ước số lớn thứ nhì

def channel_shuffle(x, groups=2):
    """
    Shuffle theo nhóm, nhưng nếu C không chia hết cho groups
    thì giữ nguyên phần dư (ví dụ kênh cuối) rồi gắn lại sau.
    """
    N, C, H, W = x.shape
    if C % groups == 0:
        # Trường hợp chuẩn
        x = x.reshape(N, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
        return x
    else:
        # ‘Để dành’ phần dư
        rem = C % groups
        C_main = C - rem
        x_main, x_rem = x[:, :C_main], x[:, C_main:]
        x_main = x_main.reshape(N, groups, C_main // groups, H, W)
        x_main = x_main.permute(0, 2, 1, 3, 4).reshape(N, C_main, H, W)
        return torch.cat([x_main, x_rem], dim=1)

def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0,2,4,3,5,1).contiguous() # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B,-1,rH,rW,C) # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out

def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous() # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out
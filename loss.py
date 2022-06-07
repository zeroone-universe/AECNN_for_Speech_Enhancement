from os import X_OK
from torch.nn import MSELoss, L1Loss
import torch 
import scipy.signal as sig
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sig
import numpy as np
from config import *


class SISNRLoss:
    def __init__(self):
        self.name = "SISNRLoss"

    def __call__(self, x_proc, x_orig):
        length = min(x_proc.shape[-1], x_orig.shape[-1])
        x_proc = x_proc[..., :length].squeeze()
        x_orig = x_orig[..., :length].squeeze()

        x_orig_zm = x_orig - x_orig.mean(dim = -1, keepdim = True)
        x_proc_zm = x_proc - x_proc.mean(dim = -1, keepdim = True)

        x_dot = torch.sum(x_orig_zm * x_proc_zm, dim = -1, keepdim = True)
        
        s_target = x_dot * x_orig_zm / (1e-10 + x_orig_zm.pow(2).sum(dim = -1, keepdim = True))

        e_noise = x_proc_zm - s_target

        SISNR = e_noise.norm(2, dim = -1) + 1e-10

        SISNR = 20*torch.log10(SISNR/(s_target.norm(2, dim = -1)+1e-10))

        return SISNR.mean()

    def get_name(self):
        return self.name


class STFTLoss:
    def __init__(self):
        self.name = "SISNRLoss"

    def __call__(self, x_proc, x_orig):
        
        total_num = x_proc.shape[0]
        total_loss = 0
        for idx in range(total_num):
            x_noisy = x_proc[idx]
            x_target = x_orig[idx]
            loss = torch.mean(torch.abs(stft_RIsum(x_target) - stft_RIsum(x_noisy)))
            total_loss+=loss

        return total_loss/total_num
    
    def get_name(self):
        return self.name


def stft_RIsum(x, nfft=STFTLOSS_WINDOW_SIZE, win_length=STFTLOSS_WINDOW_SIZE, hop_length=STFTLOSS_HOP_SIZE):
    
    window = torch.hann_window(win_length).to(x.device)

    X = torch.stft(x, nfft, hop_length=STFTLOSS_HOP_SIZE, win_length=STFTLOSS_WINDOW_SIZE,
                window = window)
    return torch.abs(X[...,0]) + torch.abs(X[...,1])

if __name__ == "__main__":
    # sisnr_loss = SISNRLoss()
    # orig = torch.randn(100, 1, 512).cuda()
    # noise = torch.randn(100, 1, 512).cuda()

    # x_dot = torch.sum(orig*noise)

    # a = sisnr_loss(orig + noise , orig)
    # print(a)


    clean = torch.rand(4,1,2048)
    noisy = torch.rand(4,1,2048)
    stft_loss = STFTLoss()
    loss = stft_loss(clean, noisy)
    print(loss)
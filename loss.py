from os import X_OK
from torch.nn import MSELoss, L1Loss
import torch 
import scipy.signal as sig
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sig
import numpy as np
from config import *

from torchaudio.transforms import MelSpectrogram


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
        self.name = "STFTLoss"

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

class MelSpecLoss:
    def __init__(self):
        self.name = "MelSpecLoss"
        self.eps = 1e-4
    def __call__(self, x_proc, x_orig):
        L=0
        for i in range(6,12):
            s = 2**i
            alpha_s = (s/2)**0.5
            melspec = MelSpectrogram(sample_rate=16000, n_fft=s, hop_length=s//4, n_mels=64, wkwargs={"device": x_orig.device}).to(x_orig.device)
            S_x = melspec(x_orig)
            
            S_G_x = melspec(x_proc)
            
            loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+self.eps)-torch.log(S_G_x.abs()+self.eps))**2).sum(dim=-2)**0.5).sum()
            L += loss
    
        return L

    def get_name(self):
        return self.name



if __name__ == "__main__":
    # sisnr_loss = SISNRLoss()
    # orig = torch.randn(100, 1, 512).cuda()
    # noise = torch.randn(100, 1, 512).cuda()

    # x_dot = torch.sum(orig*noise)

    # a = sisnr_loss(orig + noise , orig)
    # print(a)
    melspec_loss = MelSpecLoss()
    
    clean = torch.rand(4,1,2048)
    noisy = torch.rand(4,1,2048)

    loss = melspec_loss(clean, noisy)
    print(loss)
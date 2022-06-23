from os import X_OK
from torch.nn import MSELoss, L1Loss
import torch 
import torch as th
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

def Demucsstft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = th.stft(x, fft_size, hop_size, win_length, window.to(x.device))

    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return th.sqrt(th.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class DemucsSpectralConvergengeLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super().__init__()

    def forward(self, x_mag, y_mag, norm='total'):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        if norm == 'total':
            loss =  th.norm(y_mag - x_mag, p="fro") / th.norm(y_mag, p="fro")
        elif norm == 'batchwise':
            loss = th.norm(y_mag - x_mag, p="fro", dim=(1, 2)) / th.norm(y_mag, p="fro", dim=(1 , 2))
            loss = loss.mean()
        else:
            loss = F.l1_loss(x_mag, y_mag)
            
        return loss


class DemucsLogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super().__init__()

    def forward(self, x_mag, y_mag, endim = False):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        
        if endim:
            batch_avg_en  = y_mag.mean(dim = (1, 2), keepdim=True)
            batch_avg_err = (th.log(y_mag)-th.log(x_mag)).abs().mean(dim = (1, 2), keepdim=True) 
            loss = (batch_avg_en*batch_avg_err).mean()
        else:
            loss = F.l1_loss(th.log(y_mag), th.log(x_mag))
        
        return loss


class DemucsSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(th, window)(win_length))
        self.spectral_convergenge_loss = DemucsSpectralConvergengeLoss()
        self.log_stft_magnitude_loss = DemucsLogSTFTMagnitudeLoss()

    def forward(self, x, y, norm='total', endim=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = Demucsstft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = Demucsstft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag, norm=norm)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag, endim=endim)

        return sc_loss, mag_loss

class DemucsLoss(nn.Module):
    """Multi resolution STFT loss module used in DEMUCS."""

    def __init__(self,
                #  fft_sizes=[1024, 2048, 512],
                #  hop_sizes=[120, 240, 50],
                #  win_lengths=[600, 1200, 240],
                fft_sizes = [1024, 2048, 512, 256, 128, 64],
                hop_sizes = [120, 240, 50, 25, 12, 6],
                win_lengths = [600, 1200, 240, 120, 60, 30],
                 window="hann_window", factor_sc=0.5, factor_mag=0.5,
                 time_norm = 'UnNorm', freq_lin_norm = 'Total', freq_log_endim = False, 
                 **kwargs):
        
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super().__init__()
        
        self.name = 'DemucsLoss_T_{time_norm}_SC_{freq_lin_norm}_{factor_sc}_MAG_{freq_log_endim}_{factor_mag}'.format(
            time_norm = time_norm,
            freq_lin_norm = freq_lin_norm if factor_sc != 0 else 'None',
            factor_sc = factor_sc,
            freq_log_endim = 'EnDim' if freq_log_endim and factor_mag != 0 else 'NoDim',
            factor_mag = factor_mag
            )
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [DemucsSTFTLoss(fs, ss, wl, window)]
        
        assert time_norm in ['Total', 'Batchwise', 'UnNorm']
        assert freq_lin_norm in ['Total', 'Batchwise', 'UnNorm']
        
        self.time_norm = time_norm.lower()
        self.freq_lin_norm = freq_lin_norm.lower()  
        self.freq_log_endim = freq_log_endim
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y, *args, **kwargs):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        x = x[...,:y.shape[-1]].squeeze()
        y = y[...,:x.shape[-1]].squeeze()        
        
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y, norm = self.freq_lin_norm, endim = self.freq_log_endim)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        
        if self.time_norm == 'total':
            t_loss =  th.norm(x - y, p=1) / (th.norm(y, p=1)+1e-10)
        elif self.time_norm == 'batchwise':
            t_loss = th.norm(x - y, p=1, dim=-1) / (th.norm(y, p=1, dim=-1)+1e-10)
            t_loss = t_loss.mean()
        else:
            t_loss = F.l1_loss(x, y)
        
        loss  = t_loss + self.factor_sc*sc_loss + self.factor_mag*mag_loss
        return loss
    
    def get_name(self):
        return self.name
    
    def test(self):
        x = th.randn(2, 16000)
        y = th.randn(2, 16000)
        print(self(x, y))
        
    

if __name__ == "__main__":
    # sisnr_loss = SISNRLoss()
    # orig = torch.randn(100, 1, 512).cuda()
    # noise = torch.randn(100, 1, 512).cuda()

    # x_dot = torch.sum(orig*noise)

    # a = sisnr_loss(orig + noise , orig)
    # print(a)
    demucs_loss = DemucsLoss()
    
    clean = torch.rand(4,1,2048)
    noisy = torch.rand(4,1,2048)


    loss = demucs_loss(clean, noisy)
    print(loss)
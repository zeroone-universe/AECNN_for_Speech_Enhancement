import torch

class STFTLoss:
    def __init__(self, config):
        self.name = "STFTLoss"
        self.stft_risum = stft_RIsum(
            nfft = config['loss']['window_size'],
            window_size = config['loss']['window_size'],
            hop_size = config['loss']['hop_size']
        )

    def __call__(self, x_proc, x_orig):
        
        total_num = x_proc.shape[0]
        total_loss = 0
        
        for idx in range(total_num):
            x_noisy = x_proc[idx]
            x_target = x_orig[idx]
            loss = torch.mean(torch.abs(self.stft_risum(x_target) - self.stft_risum(x_noisy)))
            total_loss+=loss
   
        return total_loss/total_num
    
    def get_name(self):
        return self.name

class stft_RIsum:
    def __init__(self, nfft, window_size, hop_size):
        self.nfft = nfft
        self.window_size = window_size
        self.hop_size = hop_size
        
    def __call__(self, x):
        
        window = torch.hann_window(self.window_size).to(x.device)
        X = torch.stft(x, n_fft = self.nfft, hop_length=self.hop_size, win_length=self.window_size,
                    window = window, return_complex=True)
        
        return torch.abs(X[...,0]) + torch.abs(X[...,1])
    
    

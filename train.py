import pytorch_lightning as pl
import torch

import os
import torch.nn.functional as F
import torchaudio as ta
from loss import *
from utils import *
from AECNN import AECNN

class SETrain(pl.LightningModule):
    def __init__(self, config):
        super(SETrain, self).__init__()
        self.automatic_optimization = False
        self.config = config
        
        self.kernel_size = config['model']['kernel_size']
        self.aecnn = AECNN(kernel_size = self.kernel_size)
        
        self.criterion = STFTLoss(config = config)
        
        #optimizer & scheduler parameters
        self.initial_lr = config['optim']['initial_lr']
        self.lr_gamma = config['optim']['lr_gamma']
        
        #
        self.frame_size = config["dataset"]["frame_size"]
        self.hop_size = config["dataset"]["hop_size"]
        
        #Sample for logging
        self.data_dir = config['dataset']['data_dir']
        self.path_dir_noisy_val = config['dataset']['noisy_val']
        self.path_dir_clean_val =  config['dataset']['clean_val']
        
        self.output_dir_path = config['train']['output_dir_path']
        
        self.path_sample_noisy, self.path_sample_clean = get_one_sample_path(dir_noisy_path= os.path.join(self.data_dir, self.path_dir_noisy_val), dir_clean_path=os.path.join(self.data_dir, self.path_dir_clean_val))
        
    def forward(self,x):
        output = self.aecnn(x)
        return output


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.aecnn.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.lr_gamma, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        wav_noisy, wav_clean = batch
        wav_enh = self.forward(wav_noisy)
        
        loss = self.criterion(wav_enh, wav_clean)
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        self.log("train_loss", loss,  prog_bar = True, batch_size = self.config['dataset']['batch_size'])
        
        if self.trainer.is_last_batch:
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        wav_noisy, wav_clean = batch
        wav_enh = self.forward(wav_noisy)
        
        loss = self.criterion(wav_enh, wav_clean)

        self.log("val_loss", loss, batch_size = self.config['dataset']['batch_size'], sync_dist=True)
        
    def on_validation_epoch_end(self):
        
        sample_noisy, _  = ta.load(self.path_sample_noisy)
        sample_clean, _ = ta.load(self.path_sample_clean)
        sample_noisy = sample_noisy.to(self.device)
        sample_clean =sample_clean.to(self.device)
        
        sample_enh = self.synth_one_sample(sample_noisy)
        sample_enh = sample_enh.cpu()
        
        ta.save(f"{self.output_dir_path}/sample_{self.current_epoch}.wav", sample_enh, 16000)
        
        #My implementation is showing an error in logging audio. 
        #It seems to be either an issue with the conda environment or with the code itself.
        #If possible to resolve, please leave a comment on the issue. Thank you.
        
        # self.logger.experiment.add_audio(
        #     tag='sample/enhanced',
        #     snd_tensor = sample_enh.squeeze().detach(),
        #     global_step = self.global_step,
        #     sample_rate = 16000
        # )
        
        # self.logger.experiment.add_audio(
        #     tag='sample/clean',
        #     snd_tensor = sample_clean.squeeze().detach(),
        #     global_step=self.global_step,
        #     sample_rate = 16000
        # )
        
    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def synth_one_sample(self, wav):
        wav = wav.unsqueeze(1)
        wav_padded = F.pad(wav, (0, self.frame_size), "constant", 0)
        wav_seg = wav_padded.unfold(-1,self.frame_size, self.hop_size)
        B, C, T, L = wav_seg.shape
        
        wav_seg = wav_seg.transpose(1,2).contiguous()
        wav_seg = wav_seg.view(B*T, C, L) 

        wav_seg = self.forward(wav_seg)
        wav_seg.view(B,T,C,L).transpose(1,2).contiguous()
        wav_seg = wav_seg.view(B, C*T, L)
        
        wav_rec = F.fold(
            wav_seg.transpose(1,2).contiguous()*torch.hann_window(self.frame_size, device = wav_seg.device).view(1, -1, 1),
            output_size = [1, (wav_seg.shape[-2]-1)*self.hop_size + self.frame_size],
            kernel_size = (1, self.frame_size),
            stride = (1, self.hop_size)
        ).squeeze(-2)
        
        wav_rec = wav_rec / (self.frame_size/(2*self.hop_size))
        
        wav_rec = wav_rec.squeeze(0)
        return wav_rec
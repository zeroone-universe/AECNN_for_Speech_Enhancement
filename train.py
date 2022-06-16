from fileinput import filename
import pytorch_lightning as pl
import torch
from torch import nn

import torchaudio as ta

import sys

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

from AECNN import AECNN
from loss import *
from config import *

from pesq import pesq

class TrainAECNN(pl.LightningModule):
    def __init__(self, args):
        super(TrainAECNN, self).__init__()
        self.model = AECNN()
        self.lr = INITIAL_LR
        self.model = AECNN(in_channels=1, out_channels = 1, num_layers = NUM_LAYERS, kernel_size=KERNEL_SIZE)
        
    def forward(self,x):
        x_padded = F.pad(x, (0,WINDOW_SIZE), "constant", 0)
        x_seg =x_padded.unfold(-1, WINDOW_SIZE, HOP_SIZE)
        B, C, T, L = x_seg.shape
        x_seg = x_seg.transpose(1,2).contiguous()
        x_seg = x_seg.view(B*T, C, L)

        output_seg = self.model(x_seg)

        output = output_seg.view(B,T,C,L).transpose(1,2).contiguous()
        output = output.view(B, C*T, L)

        wav_rec = F.fold(
            output.transpose(1,2).contiguous()*torch.hann_window(WINDOW_SIZE, device = x.device).view(1, -1, 1),
            output_size = [1, (output.shape[-2]-1) *HOP_SIZE +  WINDOW_SIZE],
            kernel_size = (1,  WINDOW_SIZE),
            stride = (1, HOP_SIZE)    
        ).squeeze(-2)

        wav_rec = wav_rec[...,:x.shape[-1]]

        return x_seg, output_seg, wav_rec

    def loss_fn(self, s_noisy, s_orig):
        if LOSS_TYPE == "SISNRLoss":
            loss_function = SISNRLoss()
        elif LOSS_TYPE == "STFTLoss":
            loss_function = STFTLoss()
        elif LOSS_TYPE == "MelspecLoss":
            loss_function = MelSpecLoss()
        return loss_function(s_noisy, s_orig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_GAMMA, last_epoch = - 1, verbose=True)
        return ([optimizer], [lr_scheduler])
        
    def training_step(self, batch, batch_idx):
        wav_dist, wav_target, _ = batch
        x_seg, output_seg, wav_enh = self.forward(wav_dist)
        

        #wav_target을 segmentation
        wav_target = F.pad(wav_target, (0,WINDOW_SIZE), "constant", 0)
        wav_target_seg =wav_target.unfold(-1, WINDOW_SIZE, HOP_SIZE)
        B, C, T, L = wav_target_seg.shape
        wav_target_seg = wav_target_seg.transpose(1,2).contiguous()
        wav_target_seg = wav_target_seg.view(B*T, C, L)

        loss = self.loss_fn(output_seg, wav_target_seg)
        # loss = self.loss_fn(wav_enh, wav_target)
        self.log("training_loss" , loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        wav_dist, wav_target, filename = batch

        target_seg, output_seg, wav_enh  = self.forward(wav_dist)
        
         #wav_target을 segmentation
        wav_target = F.pad(wav_target, (0,WINDOW_SIZE), "constant", 0)
        wav_target_seg =wav_target.unfold(-1, WINDOW_SIZE, HOP_SIZE)
        B, C, T, L = wav_target_seg.shape
        wav_target_seg = wav_target_seg.transpose(1,2).contiguous()
        wav_target_seg = wav_target_seg.view(B*T, C, L)

        val_loss = self.loss_fn(output_seg, wav_target_seg)
        
        if self.current_epoch >= EPOCHS_SAVE_START:
            wav_enh_cpu = wav_enh.squeeze(0).cpu()
            ta.save(os.path.join(OUTPUT_DIR_PATH, f"{filename[0]}.wav"), wav_enh_cpu, 16000 )

        wav_target = wav_target.squeeze().cpu().numpy()
        wav_enh = wav_enh.squeeze().cpu().numpy()

        val_pesq = pesq(fs = 16000, ref = wav_target, deg = wav_enh, mode = "wb")

        self.log("val_loss", val_loss)
        self.log("val_pesq", val_pesq)


        
    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass
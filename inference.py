import argparse
import torchaudio as ta
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from train import SETrain
from AECNN_for_Speech_Enhancement.atamodule import *
from utils import *
import yaml

def inference(config, args):
    
    se_train = SETrain.load_from_checkpoint(args.path_ckpt, config = config)
    
    wav_noisy, _ =ta.load(args.path_in)
    wav_enh = synthesize(wav_noisy, se_train)
    wav_enh = wav_enh.squeeze(0).cpu()
    
    filename = get_filename(args.path_in)
    ta.save(os.path.join(os.path.dirname(args.path_in),filename[0]+"_clean"+filename[1]), wav_enh, 16000)
    

def synthesize(wav, model):
    frame_size = config["dataset"]["frame_size"]
    hop_size = config["dataset"]["hop_size"]
    
    wav = wav.unsqueeze(1)
    wav_padded = F.pad(wav, (0, frame_size), "constant", 0)
    wav_seg = wav_padded.unfold(-1,frame_size, hop_size)
    B, C, T, L = wav_seg.shape

    wav_seg = wav_seg.transpose(1,2).contiguous()
    wav_seg = wav_seg.view(B*T, C, L) 

    wav_seg = model.forward(wav_seg)
    wav_seg.view(B,T,C,L).transpose(1,2).contiguous()
    wav_seg = wav_seg.view(B, C*T, L)

    wav_rec = F.fold(
        wav_seg.transpose(1,2).contiguous()*torch.hann_window(frame_size, device = wav_seg.device).view(1, -1, 1),
        output_size = [1, (wav_seg.shape[-2]-1)*hop_size + frame_size],
        kernel_size = (1, frame_size),
        stride = (1, hop_size)
    ).squeeze(-2)
    
    wav_rec = wav_rec / (frame_size/(2*hop_size))
    return wav_rec

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ckpt", type = str)
    parser.add_argument("--path_in", type = str, help = "path of wav file or directory")
    
    args = parser.parse_args()
    
    config = yaml.load(open("./config.yaml", 'r'), Loader=yaml.FullLoader)
    
    inference(config, args)
    

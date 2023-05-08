import argparse
import torchaudio as ta
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from train import SETrain

from utils import *
import yaml

def inference(config, args):
    
    se_train = SETrain.load_from_checkpoint(args.path_ckpt, config = config)
    se_train.aecnn.eval()
    
    if args.mode == "wav":
        wav_noisy, _ =ta.load(args.path_in)
        wav_enh = se_train.synth_one_sample(wav_noisy)
        wav_enh = wav_enh.cpu()
        
        filename = get_filename(args.path_in)
        ta.save(os.path.join(os.path.dirname(args.path_in),filename[0]+"_proc"+filename[1]), wav_enh, 16000)
    
    elif args.mode == "dir":
        check_dir_exist(args.path_out)
        
        path_wavs = get_wav_paths(args.path_in)
        for path_wav in path_wavs:
            wav_noisy, _  = ta.load(path_wav)
            wav_enh = se_train.synth_one_sample(wav_noisy)
            wav_enh = wav_enh.cpu()
            ta.save(os.path.join(args.path_out, os.path.basename(path_wav)), wav_enh, 16000)

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ckpt", type = str)
    parser.add_argument("--mode", type = str, help = 'wav/dir', default = 'wav')
    parser.add_argument("--path_in", type = str, help = "path of input wav file or directory")
    parser.add_argument("--path_out", type = str, help = "path of directory of output file")
    
    args = parser.parse_args()
    config = yaml.load(open("./config.yaml", 'r'), Loader=yaml.FullLoader)
    
    inference(config, args)
    
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta
import os

import pytorch_lightning as pl

from utils import *

class SEDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
    def __init__(self, path_dir_noisy, path_dir_clean, frame_size, hop_size):
        self.path_dir_noisy   = path_dir_noisy
        self.path_dir_clean   = path_dir_clean  

        self.wavs = []

        paths_wav_noisy= get_wav_paths(self.path_dir_noisy)
        paths_wav_clean = get_wav_paths(self.path_dir_clean)

        for wav_idx, (path_wav_clean, path_wav_noisy) in enumerate(zip(paths_wav_clean, paths_wav_noisy)):
            print(f'\r{wav_idx} th file loaded', end='')
            wav_noisy, _ = ta.load(path_wav_noisy)
            wav_clean, _ = ta.load(path_wav_clean)
            
            wav_noisy_seg = segmentation(wav_noisy, frame_size, hop_size)
            wav_clean_seg = segmentation(wav_clean, frame_size, hop_size)
            
            for idx in range(wav_clean_seg.shape[0]):
                self.wavs.append([wav_noisy_seg[idx], wav_clean_seg[idx]])
    
    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.wavs)
    
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        return self.wavs[idx]
        

class SEDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.data_dir = config['dataset']['data_dir']
        
        self.path_dir_noisy_train = config['dataset']['noisy_train']
        self.path_dir_noisy_val = config['dataset']['noisy_val']
        
        self.path_dir_clean_train =  config['dataset']['clean_train']
        self.path_dir_clean_val =  config['dataset']['clean_val']

        self.frame_size = config["dataset"]["frame_size"]
        self.hop_size = config["dataset"]["hop_size"]
        
        self.batch_size = config['dataset']['batch_size']
        self.num_workers = config['dataset']['num_workers']

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.train_dataset = SEDataset(
            path_dir_noisy = os.path.join(self.data_dir, self.path_dir_noisy_train),
            path_dir_clean = os.path.join(self.data_dir, self.path_dir_clean_train),
            frame_size = self.frame_size,
            hop_size = self.hop_size
            )


        self.val_dataset = SEDataset(
            path_dir_noisy = os.path.join(self.data_dir, self.path_dir_noisy_val),
            path_dir_clean = os.path.join(self.data_dir, self.path_dir_clean_val),
            frame_size = self.frame_size,
            hop_size = self.hop_size
            )


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        pass
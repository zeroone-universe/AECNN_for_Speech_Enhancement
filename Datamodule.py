import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta
import numpy as np
import os

import pytorch_lightning as pl


from config import *
from utils import *

class CEDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
    def __init__(self, path_dir_dist, path_dir_orig, seg_len = SEG_LEN, mode = "train6"):
        self.path_dir_dist   = path_dir_dist
        self.path_dir_orig   = path_dir_orig  

        
        self.seg_len = seg_len
        self.mode = mode

        self.wavs={}
        self.filenames= []


        paths_wav_orig = get_wav_paths(self.path_dir_orig)
        paths_wav_dist= get_wav_paths(self.path_dir_dist)

        for path_wav_orig, path_wav_dist in zip(paths_wav_orig, paths_wav_dist):
            filename=get_filename(path_wav_orig)[0]
            wav_orig, self.sr = ta.load(path_wav_orig)
            wav_dist, _ = ta.load(path_wav_dist)
            self.wavs[filename]=(wav_orig, wav_dist)
            self.filenames.append(filename)
            print('\r%d th file loaded'%len(self.filenames), end='')
        self.filenames.sort()
        

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.filenames)


    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        (wav_orig, wav_dist) = self.wavs[filename]
        
        if self.seg_len > 0 and self.mode == "train":
            duration= int(self.seg_len * self.sr)

            wav_orig= wav_orig.view(1,-1)
            wav_dist= wav_dist.view(1,-1)

            sig_len = wav_orig.shape[-1]

            t_start = np.random.randint(
                low = 0,
                high= np.max([1, sig_len- duration - 2]),
                size = 1
            )[0]
            t_end = t_start + duration

            wav_orig = wav_orig.repeat(1, t_end // sig_len + 1) [ ..., t_start : t_end]
            wav_dist = wav_dist.repeat(1, t_end// sig_len + 1) [ ..., t_start : t_end]

        return wav_dist, wav_orig, filename
        


class CEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = DATA_DIR, batch_size = BATCH_SIZE, seg_len=SEG_LEN):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seg_len = seg_len

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.train_dataset = CEDataset(
            path_dir_dist = os.path.join(DATA_DIR, NOISY_TRAIN),
            path_dir_orig = os.path.join(DATA_DIR, TARGET_TRAIN),
            mode = "train"
            )


        self.test_dataset = CEDataset(
            path_dir_dist = os.path.join(DATA_DIR, NOISY_TEST),
            path_dir_orig = os.path.join(DATA_DIR, TARGET_TEST),
            seg_len = self.seg_len,
            mode = "test"
            )



    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1)
    

if __name__=="__main__":
    ce_datamodule = CEDataModule()
    ce_datamodule.setup()
    
    train_dataloader = ce_datamodule.train_dataloader()
    print(next(iter(train_dataloader)))

    test_dataloader = ce_datamodule.test_dataloader()
    print(next(iter(test_dataloader)))
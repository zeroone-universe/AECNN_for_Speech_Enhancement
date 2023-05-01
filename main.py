from train import SETrain

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodule import SEDataModule

import yaml
from utils import *

def main(args):
    pl.seed_everything(config['random_seed'], workers=True)
    se_datamodule = SEDataModule(config)
    se_train = SETrain(config)
    
    check_dir_exist(config['train']['output_dir_path'])
    check_dir_exist(config['train']['logger_path'])

    tb_logger = pl_loggers.TensorBoardLogger(config['train']['logger_path'], name=f"SE_logs")


    tb_logger.log_hyperparams(config)
    
    checkpoint_callback = ModelCheckpoint(
    filename = "{epoch}-{val_loss:.4f}",
    save_top_k = 1,
    mode = 'min',
    monitor = "val_loss"
    )

    trainer=pl.Trainer(devices=config['train']['devices'], accelerator="gpu", strategy='ddp',
    max_epochs=config['train']['total_epoch'],
    callbacks= [checkpoint_callback],
    logger=tb_logger,
    profiler = "simple"
    )

    trainer.fit(se_train, se_datamodule)

if __name__ == "__main__":

    config = yaml.load(open("./config.yaml", 'r'), Loader=yaml.FullLoader)
    main(config)
    
# python inference.py --path_ckpt "/media/youngwon/Neo/NeoChoi/Projects/AECNN_for_Speech_Enhancement/logger/SRGAN_logs/version_4/checkpoints/output.ckpt" --path_in “/media/youngwon/Neo/NeoChoi/Projects/test/noisy.wav”
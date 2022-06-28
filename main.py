import argparse
from Datamodule import CEDataModule

from train import TrainAECNN

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils import *
from config import *

def main(args):
    pl.seed_everything(RANDOM_SEED, workers=True)

    ce_datamodule = CEDataModule(data_dir = DATA_DIR, batch_size = BATCH_SIZE, seg_len = SEG_LEN)
    train_aecnn = TrainAECNN(args)
    
    check_dir_exist(OUTPUT_DIR_PATH)
    tb_logger = pl_loggers.TensorBoardLogger(LOGGER_PATH, name=f"AECNN_logs")

    checkpoint_callback = ModelCheckpoint(
    filename = "{epoch}-{val_pesq:.2f}-{val_loss:.2f}",
    verbose = True,
    save_last = True,
    save_top_k = 2,
    monitor = "val_pesq",
    mode = "max"
    )

    trainer=pl.Trainer(gpus=1,
    max_epochs=MAX_EPOCHS,
    progress_bar_refresh_rate=1,
    callbacks=[checkpoint_callback],
    logger=tb_logger,
    default_root_dir="/media/youngwon/Neo/NeoChoi/TIL/Pytorch-DL/AECNN/model_checkpoint"
    )

    trainer.fit(train_aecnn, ce_datamodule)







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Train AECNN")
    args = parser.parse_args()
    main(args)
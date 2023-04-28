# Real-time Speech Frequency Bandwidth Extension

This repository contains the unofficial pytorch lightning implementation of the model described in the paper [A New Framework for CNN-Based Speech Enhancement in the Time Domain](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8701652) by Ashutosh Pandey and Deliang Wang. 

## Requirements
 
To run this code, you will need:

- pytorch_lightning==2.0.0
- PyYAML==6.0
- torch==2.0.0
- torchaudio==2.0.0

To automatically install these libraries, run the following command:

```pip install -r requirements.txt```

## Usage

To run the code on your own machine, follow these steps:

1. Open the 'config.yaml' file and modify the file paths (and hyperparameters as needed).
2. Run the 'main.py' file to start training the model.

The trained model will be saved as ckpt file in 'logger' directory. You can then use the trained model to perform real-time speech frequency bandwidth extension on your own audio wav file by running the 'inference.py' file as

```python inference.py --path_ckpt <path of checkpoint file> --path_in <path of wav file>```

Directory-level inference is planned to be supported soon.

## Note
- 2023.4.28 The code has been modified to be compatible with PyTorch Lightning 2.0 environment! It includes support for inference as well.
- Feel free to provide issues!

## Citation

```bibtex
@ARTICLE{8701652,
  author={Pandey, Ashutosh and Wang, DeLiang},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={A New Framework for CNN-Based Speech Enhancement in the Time Domain}, 
  year={2019},
  volume={27},
  number={7},
  pages={1179-1188},
  doi={10.1109/TASLP.2019.2913512}}
```

# README

This repo provide a simple pytorch trainer for image classification, supporting multi-GPU training. The code is clean and simple, so it can be easily modifed for other tasks. Training and evaluation metrics are recorded with tensorboard, including losses, graidients, accuracy, etc.

## Requirements

First install pytorch and torchvision following [https://pytorch.org/](https://pytorch.org/), then install the following dependencies.
```bash
pip install pyyaml
pip install pandas scipy matplotlib
pip install tensorboard
pip install jupyter  # for visualization on linux server connected with vscode ssh.
```

## Training

Change parameters in `train.sh` and run it:
```bash
sh train.sh
```

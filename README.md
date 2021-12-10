# Fork of CaraNet for Hamlyn Winter School of Surgical Imaging and Vision 2021

Note: This is a fork of [CaraNet](https://github.com/AngeLouCN/CaraNet)

**Original technique report of CaraNet:** [CaraNet](https://arxiv.org/ftp/arxiv/papers/2108/2108.07368.pdf)


The fork contains changes that were done as a team effort during Hamlyn Winter School of Surgical Imaging and Vision 2021.

Topic of the the mini-project:
**Automatic Polyp Segmentation in Colon Capsule Endoscopy**

## Approach
As we could only find few test data on colon capsule endoscopy, we used cononoscopy training data and tried to modify teh training approach to adapt to colon capsule data.

Results were tested on [Kvasir-capsule](https://github.com/simula/kvasir-capsule) data set which also contains masks which we used as ground truth. 
The results were not satisfactory but our goal rather was to get familiar with the general workflow. We also tried [NanoNet](https://arxiv.org/pdf/2104.11138.pdf) and a [custom implementation](https://github.com/simongeek/CapsuleEndoscopy).

Main changes compared to CaraNet are the adaptions on augmentation using albumeration (data_loader.py ln 30) to bring the cononoscopy training data closer to capsule data.

General changelog:
* changes on python code to support windows style paths
* adding albumerations 

Great thanks for the original effort on CaraNet!

### Installation & Usage
Please refer to [CaraNet](https://github.com/AngeLouCN/CaraNet)

### Enviroment
We were running it nicely on:
- Python 3.9.7; (3.10.0 at that time wasn't supported)
- Windows 10, NVidia Quadro RTX 5000 (16GB)

### Installed packages:
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
...or choose your pytoch flavour.

On top of what is described on CaraNet:

```
pip install albumentations
```

Again, kudos to [CaraNet](https://github.com/AngeLouCN/CaraNet)



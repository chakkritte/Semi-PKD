# Semi-PKD: Semi-supervised Pseudoknowledge Distillation for saliency prediction

This paper has been published to ICT Express.

This offical implementation of Semi-PKD (Semi-Supervised Pseudoknowledge Distillation) from Semi-PKD: Semi-supervised Pseudoknowledge Distillation for saliency prediction by [Chakkrit Termritthikun](https://chakkritte.github.io/cv/).

![](https://img.shields.io/badge/-PyTorch%20Implementation-blue.svg?logo=pytorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Overview

This repository contains the source code for Semi-PKD, which accompanies the research paper titled **Semi-PKD: Semi-supervised Pseudoknowledge Distillation for saliency prediction**. The purpose of this repository is to provide transparency and reproducibility of the research results presented in the paper.

**This code is based on the implementation of  [EML-NET-Saliency](https://github.com/SenJia/EML-NET-Saliency), [SimpleNet](https://github.com/samyak0210/saliency), [MSI-Net](https://github.com/alexanderkroner/saliency), and [EEEA-Net](https://github.com/chakkritte/EEEA-Net).**

## Prerequisite for server
 - Tested on Ubuntu OS version 22.04 LTS
 - Tested on Python 3.11.8
 - Tested on CUDA 12.3
 - Tested on PyTorch 2.2.1 and TorchVision 0.17.1
 - Tested on NVIDIA RTX 4090 24 GB

### Cloning source code

```
git clone https://github.com/chakkritte/Semi-PKD/
cd Semi-PKD
mkdir data
```

## The dataset folder structure:

```
PKD
|__ data
    |_ salicon
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ mit1003
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ cat2000
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ pascals
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ osie
      |_ fixations
      |_ saliency
      |_ stimuli
```

### Creating new environments

```
conda create -n salnas python=3.11.8
conda activate salnas
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install Requirements

```
pip install -r requirements.txt --no-cache-dir
```

## Citation

If you use Semi-PKD or any part of this research, please cite our paper:
```
@article{TERMRITTHIKUN2024,
  title = {Semi-PKD: Semi-supervised Pseudoknowledge Distillation for saliency prediction},
  journal = {ICT Express},
  year = {2024},
  issn = {2405-9595},
  doi = {https://doi.org/10.1016/j.icte.2024.11.004},
  author = {Chakkrit Termritthikun and Ayaz Umer and Suwichaya Suwanwimolkul and Ivan Lee},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


# Semi-PKD: Semi-Supervised Pseudoknowledge Distillation for Saliency Prediction

**This code is based on the implementation of  [EML-NET-Saliency](https://github.com/SenJia/EML-NET-Saliency), [SimpleNet](https://github.com/samyak0210/saliency), [MSI-Net](https://github.com/alexanderkroner/saliency), and [EEEA-Net](https://github.com/chakkritte/EEEA-Net).**

## Prerequisite for server
 - Tested on Ubuntu OS version 20.04.4 LTS
 - Tested on Python 3.6.13
 - Tested on CUDA 11.6
 - Tested on PyTorch 1.10.2 and TorchVision 0.11.3
 - Tested on NVIDIA V100 32 GB (four cards)

### Cloning source code

```
git clone https://github.com/chakkritte/PKD/
cd PKD
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
conda create -n pkd python=3.6.13
conda activate pkd
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### Install Requirements

```
pip install -r requirements.txt --no-cache-dir
```

## Usage

### Training on Salicon dataset (Teacher: OFA595, Student: EEEA-C2)
```
python main.py --student eeeac2 --teacher ofa595 --dataset salicon --model_val_path model_salicon.pt
```

## Citation

If you use Semi-PKD or any part of this research, please cite our paper:
```
```

## License 

Apache-2.0 License

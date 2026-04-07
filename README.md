## Introduction

This repository contains the artifact submission for the paper ELSA accepted at ISCA 2026 (Paper ID: 124). The artifact supports reproducing the results presented in the paper, including evaluations of Spiking Neural Networks (SNNs) at each time-step and performance assessment of the ELSA accelerator.

## Environment Setup

### 1. Install Anaconda

This project requires Anaconda 3 (version 24.5.0). You can install it using the following commands:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-2023.03-Linux-x86_64.sh
```
Follow the on-screen instructions to complete the installation.

### 2. Set up the ELSA Conda Environment

Download the ELSA Conda environment from the provided link [ELSA.tar.gz](https://zenodo.org/records/19446695) and extract it:

```bash
mkdir -p ~/anaconda3/envs/ELSA
tar -xzf ELSA.tar.gz -C ~/anaconda3/envs/ELSA
source ~/anaconda3/envs/ELSA/bin/activate
```

This will activate the ELSA environment containing all dependencies required for running the artifact.


### 3. Download the checkpoints for algorithm and hardware evaluation

3.1 Download the model checkpoint for SNN accuracy and ELSA performance evaluation with url:

[Checkpoints.zip](https://zenodo.org/records/19446695)

3.2 Unzip the Checkpoints.zip and get two directories:

> Algorithm_Evaluator

> Hardware_Simulator

3.3 Put the files in `Algorithm_Evaluator` to  `~/ELSA_Algorithm/model_pool`.

3.4 Put the files in `Hardware_Simulator` to  `~/ELSA_Simluator/model_pool`, and then run `~/ELSA_Simluator/model_pool/extract_all.sh`.

After these steps, we have prepared the checkpoints for ELSA evaluation.


## ELSA Algorithm
This module evaluates the accuracy of SNNs at each time-step, as well as the accuracy and latency of elastic inference.

### ELSA Algorithm Artifact

#### Introduction

This project evaluates the accuracy of the SNN models used in ELSA, as well as the latency of elastic inference.

We evaluate multiple SNN models on **ImageNet**, **CIFAR-10**, and **CIFAR-100**.  
The corresponding checkpoints are provided at the following link:

**Checkpoint Download Link:** https://zenodo.org/records/19446695

#### Reported Accuracy

| Model | Dataset | ANN Accuracy | SNN Accuracy | Elastic SNN Accuracy | Latency Reduction |
|------|---------|--------------|--------------|----------------------|----------------------|
| VGG16 | CIFAR-10 | 91.57 | 91.48 | N/A |N/A |
| VGG16 | CIFAR-100 | 73.94 | 73.85 | N/A |N/A |
| ResNet18 | ImageNet | 67.611 | 67.537 | 67.27 (confidence threshold=0.3) | 14.76% |
| ResNet34 | ImageNet | 71.512 | 71.558 | 68.36 (confidence threshold=0.3) | 23.31% |
| ResNet50 | ImageNet | 74.842 | 74.776 | 73.40 (confidence threshold=0.3) | 19.90% |
| ViT | ImageNet | 78.398 | 78.66 | 78.39 (confidence threshold=0.8) | 11.50% |

#### 1. VGG16

This part mainly evaluates the accuracy of the VGG16 model on CIFAR-10 and CIFAR-100.

##### VGG16 on CIFAR-10

Go to the directory `~/ELSA_Algorithm/VGG` and Run the following script:

```bash
bash run_snn_bn_vgg16_cifar10.sh
```

##### VGG16 on CIFAR-100

Go to the directory `~/ELSA_Algorithm/VGG` and run the following script:

```bash
bash run_snn_bn_vgg16_cifar100.sh
```

####  2. ResNet

This part mainly evaluates the accuracy of ResNet18, ResNet34, and ResNet50 on ImageNet.

### Full Inference

Go to the directory `~/ELSA_Algorithm/ResNet` and run the following scripts:

```bash
bash run_snn_bn_resnet18.sh
bash run_snn_bn_resnet34.sh
bash run_snn_bn_resnet50.sh
```

##### Elastic Inference

Go to the directory `~/ELSA_Algorithm/ResNet` and run the following scripts:
```bash
bash run_snn_bn_resnet18_elastic.sh
bash run_snn_bn_resnet34_elastic.sh
bash run_snn_bn_resnet50_elastic.sh
```

3. Vision Transformer
##### Full Inference

Go to the directory `~/ELSA_Algorithm/ViT` and run the following script:

```bash
bash ./scripts/vit-small_SNN_16.sh
```

##### Elastic Inference

Run the following script:

```bash
bash ./scripts/vit-small_SNN_16_elastic.sh
```




## ELSA Simulator
This module evaluates the performance of the ELSA accelerator, including energy consumption, latency, and area. It reproduces the results shown in Figure 16 and Figure 17 of the paper.

### ELSA Simulator

#### Introduction

This project provides an end-to-end, cycle-level simulator for ELSA.
The simulator supports two execution modes:

#### Slow Path

### 1. Obtain the Quantized Model Checkpoint

> Noticing: You should finish the step-3 in experiment setup.

After downloading, the model path are specified in:

`~\ELSA_Simulator\convolution\configs\elsa_models.yaml`

You may also modify this path yourself if needed.

##### 2. Reproduce Figure 16

Run the following command:

```bash
python3 run_figure16.py
```
##### 3. Reproduce Figure 17

Run the following command:

```bash
python3 run_figure17.py
```


#### Fast Path

We provide pre-generated tracer files in:

`/home/kang_you/ELSA_Simulator/tracer_files`


Users can use these tracer files to directly reproduce the results of Figure 16 and Figure 17 without running full simulations.

##### 1. Reproduce Figure 16

```bash
python3 run_figure16.py --cache-dir tracer_files --skip-sim
```

##### 2. Reproduce Figure 17

```bash
python3 run_figure17.py --cache-dir tracer_files --skip-sim
```

#### Figure 16

![Figure 16](./ELSA_Simluator/Figures/QANN_benchMark_Comparison.png)

#### Figure 17

![Figure 17](./ELSA_Simluator/Figures/SNN_benchMark_Comparison.png)

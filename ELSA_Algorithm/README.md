# ELSA Algorithm Artifact

## Introduction

This project evaluates the accuracy of the SNN models used in ELSA, as well as the latency of elastic inference.

We evaluate multiple SNN models on **ImageNet**, **CIFAR-10**, and **CIFAR-100**.  
The corresponding checkpoints are provided at the following link:

**Checkpoint Download Link:** https://zenodo.org/records/19446695

## Reported Accuracy

| Model | Dataset | ANN Accuracy | SNN Accuracy | Elastic SNN Accuracy | Latency Reduction |
|------|---------|--------------|--------------|----------------------|----------------------|
| VGG16 | CIFAR-10 | 91.57 | 91.48 | N/A |N/A |
| VGG16 | CIFAR-100 | 73.94 | 73.85 | N/A |N/A |
| ResNet18 | ImageNet | 67.611 | 67.537 | 67.27 (confidence threshold=0.3) | 14.76% |
| ResNet34 | ImageNet | 71.512 | 71.558 | 68.36 (confidence threshold=0.3) | 23.31% |
| ResNet50 | ImageNet | 74.776 | 74.776 | 73.40 (confidence threshold=0.3) | 19.90% |
| ViT | ImageNet | 78.398 | 78.66 | 78.39 (confidence threshold=0.8) | 11.50% |

## 1. VGG16

This part mainly evaluates the accuracy of the VGG16 model on CIFAR-10 and CIFAR-100.

### VGG16 on CIFAR-10

Run the following script:

```bash
bash  ~/ELSA_Algorithm/VGG/run_snn_bn_vgg16_cifar10.sh
```

### VGG16 on CIFAR-100

Run the following script:

```bash
bash ~/ELSA_Algorithm/VGG/run_snn_bn_vgg16_cifar100.sh
```

##  2. ResNet

This part mainly evaluates the accuracy of ResNet18, ResNet34, and ResNet50 on ImageNet.

### Full Inference

Run the following scripts:

```bash
bash ~/ELSA_Algorithm/ResNet/run_snn_bn_resnet18.sh
bash ~/ELSA_Algorithm/ResNet/run_snn_bn_resnet34.sh
bash ~/ELSA_Algorithm/ResNet/run_snn_bn_resnet50.sh
```

### Elastic Inference

Run the following scripts:
```bash
bash ~/ELSA_Algorithm/ResNet/run_snn_bn_resnet18_elastic.sh
bash ~/ELSA_Algorithm/ResNet/run_snn_bn_resnet34_elastic.sh
bash ~/ELSA_Algorithm/ResNet/run_snn_bn_resnet50_elastic.sh
```

3. Vision Transformer
### Full Inference

Run the following script:

```bash
bash ~/ELSA_Algorithm/ViT/scripts/vit-small_SNN_16.sh
```

### Elastic Inference

Run the following script:

```bash
bash ~/ELSA_Algorithm/ViT/scripts/vit-small_SNN_16_elastic.sh
```


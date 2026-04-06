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

Download the ELSA Conda environment from the provided link (ELSA.tar.gz) and extract it:

```bash
mkdir -p ~/anaconda3/envs/ELSA
tar -xzf ELSA.tar.gz -C ~/anaconda3/envs/ELSA
source ~/anaconda3/envs/ELSA/bin/activate
```

This will activate the ELSA environment containing all dependencies required for running the artifact.

## File Structure
### ELSA Algorithm
This module evaluates the accuracy of SNNs at each time-step, as well as the accuracy and latency of elastic inference.
### ELSA Simulator
This module evaluates the performance of the ELSA accelerator, including energy consumption, latency, and area. It reproduces the results shown in Figure 16 and Figure 17 of the paper.


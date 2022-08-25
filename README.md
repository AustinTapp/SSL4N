# SSL4N
Self/Semi-Supervised Learning for Neonate Brain Segmentation

This repository is primiarly developed for the use of self and semi supervised learning for the eventual segmentation of gray and white matter from neonate MRI volumes.

The stages of the project are as follows:
1) Completely self-supervised learning for pre-training. Reconstruction of MR volumes (T1, T2, and others) using swinUNTER (https://arxiv.org/abs/2201.01266) and a Vit backbone (https://arxiv.org/abs/2010.11929). This training uses data from: .
2) Supervised learning on Alzheimer's MR imaging to segement gray and white matter and CSF using data from ADNI (https://adni.loni.usc.edu/).
3) Fine-tuning supervised learning with a private dataset of 16 expertly segmented neonates.
4) Segmentation prediction on an unlabled dataset of 300 neonates. Deployed using Rhino Health's (https://www.rhinohealth.com/) federated learning platform. 

Built primarily with MONAI (https://github.com/Project-MONAI/MONAI) modules and pytorch lightning (https://www.pytorchlightning.ai) backend.

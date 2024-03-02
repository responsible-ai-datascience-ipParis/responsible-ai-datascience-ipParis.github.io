+++
title = 'Measuring the Transferability of Pre-trained Models: a link with Neural Collapse Distances on Target Datasets'
date = 2024-01-08T11:26:03+01:00
draft = false
+++

**Authors** : Marion Chadal and Julie Massé

This blog post discusses the paper "How Far Pre-trained Models Are from Neural Collapse on the Target Dataset Informs their Transferability" [[1]](#ref1).

# Reproducability

To reproduce their main experiment, the authors' code available on a [Github](https://github.com/BUserName/NCTI/tree/main) repository was used. A first encountered issue was the required `torch` and `torchvision` versions, which are quite old, and thus not always available to install, which was the case here. Fortunately, the  most recent versions were compatible with the code. A `requirements.txt` file would have been welcome.

A second issue is that there are remaining personal paths in some scripts, which should be replaced by downloading paths to PyTorch source models. As a consequence, the loading method from `torch` should also be replaced.

Other issues considering the datasets loading remained unsolved.

After these modifications, it is possible to run the authors' experiments on the CIFAR10 dataset. Consisting of 60 000 32x32 colour images in 10 classes, this dataset is broadly used in benchmarks for image classification. 

A Github repository with all the necessary modifications from the original code is at your disposal [here](https://github.com/marionchadal/NCTI).

# References

<a id="ref1"></a>1. Z. Wang Y.Luo, L.Zheng, Z.Huang, M.Baktashmotlagh (2023), How far pre-trained models are from neural collapse on the target dataset informs their transferabilityWang, ICCV.


<hr></hr>

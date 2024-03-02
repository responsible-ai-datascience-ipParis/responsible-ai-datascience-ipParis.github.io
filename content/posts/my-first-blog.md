+++
title = 'Measuring the Transferability of Pre-trained Models: a link with Neural Collapse Distances on Target Datasets'
date = 2024-01-08T11:26:03+01:00
draft = false
+++

**Authors** : Marion Chadal and Julie Massé

This blog post discusses the paper "How Far Pre-trained Models Are from Neural Collapse on the Target Dataset Informs their Transferability" [[1]](#ref1). It provides an explanation of it so that you can understand the usefulness of measuring transferability, and a reproduction of the authors' experiment so that you can better visualize their methodology.

# Transferability and Neural Collapse

Transferability caracterizes the ability of pre-trained models to run on downstream tasks without performing fine-tuning, but achieving comparable results. Models that exhibit high transferability are those that have learned generalizable features during pre-training—features that are not overly specific to the training data but that capture universal patterns or structures present across different datasets and domains.

Neural Collapse happens when training beyond 0 training error, i.e training error is at 0 while pushing training loss approaching 0 even further down. Imagine training a deep neural network on a dataset for a classification task. As the training process nears its end—particularly when the model is trained to a point of perfect or near-perfect classification accuracy on the training data. Intuitively, one would expect a highly overfitted and noisy model. Instead, a remarkable simplification occurs in the way the model represents the data, as it was shown in [[2]](#ref2). This training approach offers better generalization performance, better robustness, and better interpretability.

# Measuring transferability

Fine-tuning pre-trained models works as follows. First, you pick a downstream task, for which you have at your disposal several pre-trained models candidates. Your want to compare their performances to pick the best one on test set, with the optimal fine-tuning configuration. Pre-trained models are obtained by training on huge amounts of data, which require heavy time and computational resources. Then, you have to fine-tune each of them, which means that the model is further trained, but this time on a smaller dataset. Fine-tuning aims at adjusting weights and biases of the pre-trained model to your specific task. Even if the dataset to train on is smaller, you have to repeat it for all your models candidates, and one does not want that.

Transferability estimation arises as a solution to anticipate and avoid unnecessary fine-tuning, by ranking the performances of pre-trained models on a downstream task without any fine-tuning. Having a benchmark on the pre-trained models' transferability would allow you to pick the relevant ones for your own downstream task.

# Experiment

To reproduce their main experiment, the authors' code available on a [Github](https://github.com/BUserName/NCTI/tree/main) repository was used. A first encountered issue was the required `torch` and `torchvision` versions, which are quite old, and thus not always available to install, which was the case here. Fortunately, the  most recent versions were compatible with the code. A `requirements.txt` file would have been welcome.

A second issue is that there are remaining personal paths in some scripts, which should be replaced by downloading paths to PyTorch source models. As a consequence, the loading method from `torch` should also be replaced.

Other issues considering the datasets loading remained unsolved.

After these modifications, it is possible to run the authors' experiments on the CIFAR10 dataset for the group of supervised pre-trained models. Consisting of 60 000 32x32 colour images in 10 classes, this dataset is broadly used in benchmarks for image classification. 12 pre-trained models were ran on CIFAR10 to establish a ranking based on their performances in terms of NCTI available below. 

<table style="width:100%; border-collapse: collapse;" border="1">
  <thead>
    <tr>
      <th style="text-align:left; padding: 8px;">Model</th>
      <th style="text-align:left; padding: 8px;">NCTI Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px;">resnet152</td>
      <td style="padding: 8px;">2.0</td>
    </tr>
    <tr>
      <td style="padding: 8px;">resnet101</td>
      <td style="padding: 8px;">1.799</td>
    </tr>
    <tr>
      <td style="padding: 8px;">densenet201</td>
      <td style="padding: 8px;">1.434</td>
    </tr>
    <tr>
      <td style="padding: 8px;">densenet169</td>
      <td style="padding: 8px;">1.146</td>
    </tr>
    <tr>
      <td style="padding: 8px;">resnet34</td>
      <td style="padding: 8px;">0.757</td>
    </tr>
    <tr>
      <td style="padding: 8px;">resnet50</td>
      <td style="padding: 8px;">0.709</td>
    </tr>
    <tr>
      <td style="padding: 8px;">densenet121</td>
      <td style="padding: 8px;">0.655</td>
    </tr>
    <tr>
      <td style="padding: 8px;">mnasnet1_0</td>
      <td style="padding: 8px;">0.031</td>
    </tr>
    <tr>
      <td style="padding: 8px;">googlenet</td>
      <td style="padding: 8px;">-0.251</td>
    </tr>
    <tr>
      <td style="padding: 8px;">mobilenet_v2</td>
      <td style="padding: 8px;">-0.444</td>
    </tr>
    <tr>
      <td style="padding: 8px;">inception_v3</td>
      <td style="padding: 8px;">-0.732</td>
    </tr>
  </tbody>
</table>

Then, we evaluated the transferability of the supervised pre-trained models, in terms of weighted Kendall' τ, and obtained the exact same result as the one presented in the paper: 0.843.

It was not possible for us to run the experiment on the group of self-supervised pre-trained models as the authors' code included personal paths, and we were not able to find them online.

A Github repository with all the necessary modifications from the original code is at your disposal [here](https://github.com/marionchadal/NCTI).

# References

<a id="ref1"></a>1. Z. Wang Y.Luo, L.Zheng, Z.Huang, M.Baktashmotlagh (2023), How far pre-trained models are from neural collapse on the target dataset informs their transferabilityWang, ICCV.
<a id="ref2"></a>2. V. Papyana,1 , X. Y. Hanb,1 , and D.L. Donoho (2020), Prevalence of neural collapse during the terminal phase of deep learning training, National Academy of Sciences.

<hr></hr>

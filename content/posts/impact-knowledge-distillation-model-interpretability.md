+++
title = 'Knowledge Distillation:  Boosting Interpretability in Deep Learning Models'
date = 2025-03-15T12:16:21+01:00
draft = false
+++

<style
TYPE="text/css">

code.has-jax {font:inherit;
font-size:100%;
background: inherit;
border: inherit;}
</style>

<script
type="text/x-mathjax-config">

MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script','noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});

MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});

</script>
<script
type="text/javascript"
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

<!DOCTYPE html>
<html lang="fr">
<head> <meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>

<h1 style="font-size: 36px;">On the Impact of Knowledge Distillation for Model Interpretability</h1>

<h1 style="font-size: 24px;">Enhancing Trust in AI Through Transparent Student Models</h1>

<h1 style="font-size: 18px;">Authors: Bryan Chen, Rémi Calvet<br />
Paper: <a href="https://arxiv.org/abs/2305.15734">ICML 2023</a> by Hyeongrok Han, Siwon Kim, Hyun-Soo Choi, Sungroh Yoon</h1>

<!-- <p>
<i>Blog post written for the course "Recent Advances in Responsible AI" of the Master 2 Data Science, IP Paris.</i>
</p> -->

## NOUVEAU A PARTIR D'ICI

Knowledge distillation is a method to transfer the knowledge of a "teacher" model to a "student" model and can be used to get a highly performant small model from a large model. Several paper investigate why knowledge distillation improves the performance of models. This blog post highlights an other advantage of knowledge distillation that is better interpretability of models after knowledge distillation. We focus on the paper "On the Impact of Knowledge Distillation for Model Interpretability" by Hyeongrok Han and al.


## How to define interpretability ?
The first thing to know is that there are different approach to define and measure interpretability in machine learning.

For image classification, the authors choose a network dissection approach. The idea is to compare activation maps and see if areas with high activation correspond to an object or a concept on the image.

Feed a neural network model an image, pick a deep layer and count the number of neurons that detects a concept like "cat" or "dog".
We call those neurons concept detectors and will define them more precisely. In this blog post, the number of concept detectors will be the primary metric to define the interpretability of a model, the higher the more we will consider it interpretable.

Here is the pseudo code of the procedure to compute the number of concept detectors :

1) Feed an image in the neural network.
2) Choose a layer l deep in the network.
3) We suppose that the layer l contains N neurons and for each neuron indexed by i, we collect the activation maps Aᵢ(x) ∈ ℝᵈˣᵈ, where d < n.
4) Let note a_i the distribution of individual unit activation. You can think of it as an empirical distribution where the varying element is the image x. Each image contributes to construct a_i.
5) For each neuron we compute a threshold T_i such that P(a_i >= T_i) = 0.005.
6) We reshape A_i to match the dimension of the image of shape (n,n) to be able to make comparisons.
7) We create a binary mask of shape(n,n) where each element is 1 if its correspond in A_i(x) is greater or equal than the threshold T_i. This permits to considerate only the top 0.5% activation.
8) We are provided an other mask





1:  N ← the number of convolutional units in the fourth layer of f <br>
2:  for x ∈ ℝⁿˣⁿ in X do<br>
3:      for i = 1, 2, ..., N do<br>
4:          Collect the activation map Aᵢ(x) ∈ ℝᵈˣᵈ, where d < n<br>
5:      end for<br>
6:  end for<br>
7:  aᵢ ← the distribution of individual unit activation<br>
8:  for x ∈ ℝⁿˣⁿ in X do<br>
9:      for i = 1, 2, ..., N do<br>
10:         Calculate Tᵢ to satisfy P(aᵢ ≥ Tᵢ) = 0.005<br>
11:         Interpolate Aᵢ(x) to be ∈ ℝⁿˣⁿ<br>
12:         Aᵢ(x) ← Aᵢ(x) ≥ Tᵢ<br>
13:         M𝑐(x) ← annotation mask of x for concept c<br>
14:         Compute IoUᵢ,𝑐 value between Aᵢ(x) and M𝑐(x)<br>
15:         if IoUᵢ,𝑐 ≥ 0.05 then<br>
16:             Unit i is the concept detector of the concept c<br>
17:         end if<br>
18:     end for<br>
19: end for<br>






<p align="center">
  <img src="/images/Bryan_Remi/comparisons_dog.png" alt="test123">
</p>








## EN DESSOUS C'EST L'ANCIEN TRUC

## Context and Motivations




In many Deep Learning approaches, <b>Knowledge Distillation (KD)</b> is used to transfer information from a large “teacher” model to a smaller “student” model. Historically, this technique has mainly been used to improve the accuracy of the smaller model and reduce its computational cost. However, the paper <b>“On the Impact of Knowledge Distillation for Model Interpretability”</b>, published by Hyeongrok Han, Siwon Kim, Hyun-Soo Choi, and Sungroh Yoon (2023), brings forth a new perspective: beyond boosting performance, knowledge distillation can also <b>enhance the interpretability</b> of the student model.

In the realm of Responsible AI, building more transparent and understandable models is a major objective: better insight into how a neural network arrives at its predictions helps detect errors and build confidence in these systems. This is precisely the motivation behind the study summarized here.

## Paper Summary

Han et al. (2023) aim to empirically show that:

1. **Knowledge Distillation (KD)** not only increases the accuracy of student models but
2. **Also transfers “class-similarity” information** that makes the model’s activations more object-centric, thus more interpretable.

### Key Points

- **Interpretability Concept**: The authors rely on the notion of “object-centric activation” to evaluate whether a network focuses on the core regions of an object rather than its background.
- **Interpretability Measurement**: They use <b>Network Dissection</b> (Bau et al., 2017) to quantify the number of <i>concept detectors</i> within a network. The more such detectors clearly align with concepts (e.g., dog, cat, table, wheels, etc.), the more interpretable the model is considered.
- **Comparison: KD vs. Label Smoothing**: They distinguish between <i>logit distillation</i> (where the student receives the smoothed teacher outputs) and "Label Smoothing" (where the one-hot label is replaced by a uniform distribution). Results suggest that label smoothing (LS) removes class-similarity information and harms interpretability, while KD preserves or even amplifies it.
- **Extensive Experiments**:
  - On ImageNet and other datasets, they observe a marked increase in object detectors (and more focused activation) for distillation-based student models compared to models trained from scratch.
  - On synthetic data, they quantify interpretability using various metrics (five-band-scores, DiffROAR, etc.) with ground-truth heatmaps, again confirming gains from KD.
  - They replicate the phenomenon across multiple distillation methods (AT, CRD, Self-KD...) and even in other domains (NLP), lending robustness to the conclusion.

Overall, they conclude that <b>Knowledge Distillation improves both accuracy and interpretability</b> of the student model, primarily due to the <i>explicit transfer of class-similarity information</i> from the teacher.

## Key theoretical points

1. **Knowledge Distillation (KD)**
   Classic distillation (Hinton et al., 2015) consists in minimizing the distance between the outputs of the student and those of a larger teacher network:
   \[
     \mathcal{L}_{KD} = (1-\alpha)\,\text{CE}\bigl(\sigma(z_s), y\bigr) \;+\; \alpha\,T^2\,\text{CE}\bigl(\sigma(z_s^T), \sigma(z_t^T)\bigr),
   \]
   where \(\sigma\) is the softmax function, \(z_s\) and \(z_t\) are the student and teacher logits, respectively, \(\alpha\) is a mixing hyperparameter, and \(T\) is the temperature.

2. **Label Smoothing (LS)**
   In contrast, label smoothing replaces the one-hot label with a mixture of one-hot and a uniform distribution. Even if this approach can help with generalization and minor robustness, it <b>erases</b> part of the fine-grained class-similarity. The paper demonstrates that losing that similarity leads to lower interpretability (the network may latch onto less specific cues like backgrounds or contexts).

3. **Improving Interpretability**
   The authors explain that the teacher’s logits contain <b>class-similarity information</b>. For instance, if the teacher sees a “Border Collie,” its output will also give a relatively high probability for “German Shepherd” or other dog breeds. When the student model distills these logits, it picks up a richer, more generic representation for “dog,” ultimately yielding object-centric activations (including the head, legs, tail, etc.). This, in turn, aligns more closely with the target object region and thus improves interpretability.

---

## Main Experiments and Results

### a) Network Dissection on ImageNet

- **Object Detectors**: The total number of concept detectors for objects increases significantly under KD compared to a model trained from scratch.
- **Scene vs. Object**: Under Label Smoothing, one sees the reverse: a drop in the number of object detectors, offset by an increase in scene detectors. The network focuses more on the background scene than on the object itself.

### b) Synthetic Dataset

They also devised an artificial dataset in which each sample is composed of three zones: a background, a region indicating object location, and a more distinctive region. Having a <b>ground-truth heatmap</b> allows them to compute pixel-based precision/recall. These metrics confirm that the distillation-trained student focuses more strongly on the "correct" areas than a model trained from scratch.

### c) Other Interpretability Measures

They also evaluate:

- **DiffROAR Score** (Shah et al., 2021), which measures how well model attributions highlight discriminative features.
- **Gradient Perception** (inspired by Tsipras et al., 2018) to see whether gradients align with the edges or relevant zones.

In all cases, the distilled model appears more in tune with the semantically significant aspects.

### d) NLP

Finally, to demonstrate that the effect is not vision-specific, they replicate similar experiments for a sentiment classification task in NLP (a 12-layer BERT as teacher vs. a 3-layer BERT as student). Again, they find that the distilled model is “more explainable” using word-level attributions tied to positive/negative sentiment.

---

## Tutorial: How to Reproduce Part of Their Experiments

Below is a brief guide to replicate (partially) one of the experiments from the paper, highlighting important points and potential pitfalls.

### 1. Installation and Setup

1. **Obtain a “teacher” model**: For instance, a ResNet-34 pretrained on ImageNet.
2. **Initialize a “student” model**: For example, a ResNet-18 with the same input dimensions.
3. **Install Network Dissection** (or an equivalent) to evaluate interpretability via <i>concept detectors</i>.

   - The original <i>Network Dissection</i> code is available on GitHub (Bau et al., 2017).
   - You also need the <b>Broden</b> dataset, with pixel-level concept annotations.

### 2. Setting Up Distillation

1. **Formulate the Loss**:
   \[
     \mathcal{L}_{KD} = (1-\alpha)\,\text{CE}(\sigma(z_s), y) \;+\; \alpha\,T^2\,\text{CE}(\sigma(z_s^T), \sigma(z_t^T)),
   \]
   - Choose your hyperparameters \(\alpha\) (e.g., 0.1, 0.5) and temperature \(T\) (e.g., 4) based on your resources and experiments.
2. **Training Loop**:
   - For each batch, compute cross-entropy between the student and the one-hot labels, and then cross-entropy between the student and the teacher’s smoothed logits (\(\sigma(z_t^T)\)).

### 3. Evaluating Interpretability

1. **Freeze the model**: after training is complete, freeze all student model weights.
2. **Feed Broden dataset**: record the unit-level activation (feature maps in each convolutional layer) and figure out the top-0.5% threshold value.
3. **Compute IoU**: for each concept mask in Broden, measure the Intersection-over-Union between the masked activation map and the ground-truth mask.
4. **Count concept detectors**: a detector is valid if its IoU with a concept is above a threshold (e.g., 0.05).
5. **Compare**: we can specifically track object/scene/material/etc. detectors.

### 4. Challenges / Caveats

- **Hyperparameters**: The choice of \(\alpha\) and \(T\) influences both accuracy and the amount of class-similarity transmitted. A very high \(T\) can cause training instability, and a large \(\alpha\) can degrade performance if the teacher is unreliable.
- **Dataset Size**: Training on ImageNet is resource-intensive (GPU, memory). You may want to test on a smaller subset or another smaller dataset (e.g., CIFAR) first.
- **Network Dissection**: The Broden dataset is large. Make sure to have enough disk space. The IoU computation must be carefully implemented (resizing, top-k threshold, etc.).
- **Reproducibility**: Version mismatches in PyTorch or random seeds can lead to slight variations in the final counts of concept detectors. The paper often trains multiple runs from different seeds and averages results.

---

## Conclusion

This paper significantly contributes to Responsible AI by showing that knowledge distillation is not only useful for compressing large networks into high-performance smaller ones — it also <b>increases the interpretability of the student model</b>. Their conclusion is based on numerous experiments (Network Dissection, DiffROAR, etc.) and holds across various scenarios (vision, NLP, synthetic data…).

### Key Takeaways

- <b>"Class-similarity" is the linchpin</b>: The teacher’s logits are not uniform and carry critical signals about how classes relate to each other.
- <b>Label Smoothing vs. Distillation</b>: LS “flattens” the distribution, sacrificing structure; distillation, by contrast, pushes the student to focus on the object more effectively.
- <b>Real-World Impact</b>: Improved interpretability helps detect biases and mistakes more easily.
- <b>Further Work</b>: The authors discuss robustness, out-of-distribution generalization, easier debugging, and so forth. Future directions include studying self-distillation or combining other forms of interpretability.

In short, <i>On the Impact of Knowledge Distillation for Model Interpretability</i> is an excellent illustration of the Responsible AI philosophy: going beyond mere performance to prioritize transparency and trustworthiness. We hope this blog post inspires you to explore and replicate these experiments yourself, so you can better appreciate the value of “interpretable distillation”!

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Han, H., Kim, S., Choi, H.-S., & Yoon, S. (2023). <b>On the Impact of Knowledge Distillation for Model Interpretability</b>. arXiv:2305.15734.
- Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). Network dissection: Quantifying interpretability of deep visual representations. CVPR.
- Shah, H., Jain, P., & Netrapalli, P. (2021). Do input gradients highlight discriminative features? NeurIPS.
- Tsipras, D. et al. (2018). Robustness May Be at Odds with Accuracy. arXiv:1805.12152.

<br />
<i>For questions or comments, feel free to contact Bryan Chen and Rémi Calvet.</i>
Bonjour
</html>
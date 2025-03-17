+++
title = 'Knowledge Distillation:  Boosting Interpretability in Deep Learning Models'
date = 2025-03-15T12:16:21+01:00
draft = false
+++

<style TYPE="text/css">
   code.has-jax {font:inherit;
                  font-size:100%;
                  background: inherit;
                  border: inherit;}
</style>

<script TYPE="text/x-mathjax-config">
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

<script TYPE="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full">
</script>

<!DOCTYPE html>
<html lang="fr">

<head>
   <meta charset="UTF-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>

<h1 style="font-size: 28px;">Interpretability, the hidden power of knowledge distillation</h1>

<p>Published March 15, 2025</p>

<div>
  <a href="https://github.com/BryanBradfo/responsible-ai-datascience-ipParis.github.io" class="btn" style="text-decoration: none; display: inline-block; padding: 8px 16px; background-color: #f1f1f1; border: 1px solid #ddd; border-radius: 4px; color: black;">Update on GitHub</a>
</div>

<div style="display: flex; margin-top: 20px;">
   <div style="display: flex; align-items: center; margin-right: 20px;">
      <img src="/images/Bryan_Remi/bryan.jpeg" alt="Bryan Chen" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">
      <div>
         <a href="https://github.com/BryanBradfo" style="text-decoration: none; color: #0366d6;">bryanbradfo</a>
         <p style="margin: 0;">Bryan Chen</p>
      </div>
   </div>

   <div style="display: flex; align-items: center; margin-right: 20px;">
      <img src="/images/Bryan_Remi/remi.jpg" alt="Rémi Calvet" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">
      <div>
         <a href="https://github.com/RemiCSK" style="text-decoration: none; color: #0366d6;">remicsk</a>
         <p style="margin: 0;">Rémi Calvet</p>
      </div>
   </div>

</div>

Knowledge distillation is a powerful technique to transfer the knowledge from a large "teacher" model to a "student" model. While it's commonly used to improve performance and reduce computational costs by compressing large models, this blog post explores a fascinating discovery: knowledge distillation can also significantly enhance model interpretability. We'll dive into the paper <a href="https://arxiv.org/abs/2305.15734">On the Impact of Knowledge Distillation for Model Interpretability"</a> (ICML 2023) by H. Han et al., which sheds light on this novel perspective.

<p align="center">
  <img src="/images/Bryan_Remi/better_meme.png" alt="Introduction" style="width: 60%; max-width: 500px; height: auto;">
</p>

Interpretability in AI allows researchers, engineers, and decision-makers to trust and control machine learning models. Recent models show impressive performance on many different tasks and often rely on deep learning models. Unfortunately, deep learning models are also know for the difficulty to interprete them and understand how they come to a result wich can be problematic in highly sensitive applications like autonomous driving or healthcare. The article we present in this blog shows that knowledge distillation can improve the interpretability of deep learning models.

<!--
Interpretability in AI allows researchers, engineers, and decision-makers to trust and control deep learning models. When we understand how an AI model reaches its conclusions, we can debug issues faster, ensure fairness, and prevent catastrophic failures. Without interpretability, AI systems become black boxes, leaving us blind to potential biases or misjudgments. -->
<!--
<p align="center"> <img src="/images/Bryan_Remi/ai_interpretable.gif" alt="Feeling strong with interpretable AI" style="width: 60%; max-width: 500px; height: auto;"> </p> -->

When AI is a black box, you're just hoping for the best. But when you understand it, you become unstoppable.

<h2 style="font-size: 21px; display: flex; align-items: center;"> 0. Table of Contents </h2>

- [I. Crash Course on Knowledge Distillation and Label Smoothing](#i-crash-course-on-knowledge-distillation)
- [II. Defining Interpretability Through Network Dissection](#ii-defining-interpretability-through-network-dissection)
- [III. Why Knowledge Distillation Enhances Interpretability](#iii-why-knowledge-distillation-enhances-interpretability)
- [IV. Experimental Results and Reproduction](#iv-experimental-results-and-reproduction)
- [V. Beyond Network Dissection: Other Interpretability Metrics](#v-beyond-network-dissection-other-interpretability-metrics)
- [Conclusion](#conclusion)
- [Join the Discussion](#join-the-discussion)

<h2 id="i-crash-course-on-knowledge-distillation" style="font-size: 21px; display: flex; align-items: center;"> I. Crash Course on Knowledge Distillation and Label Smoothing </h2>

### What is Knowledge Distillation?

<p align="center">
  <img src="/images/Bryan_Remi/knowledge_distillation.png" alt="Knowledge Distillation Overview" style="width: 70%; max-width: 500px; height: auto;">
</p>

<a href="https://arxiv.org/pdf/1503.02531">Knowledge distillation (KD)</a> is a model compression technique introduced by Hinton et al. (2015) that transfers knowledge from a complex teacher model to a simpler student model. Unlike traditional training where models learn directly from hard labels (one-hot encodings), KD allows the student to learn from the teacher's soft probability distributions.

### The Key Mechanics of Knowledge Distillation

The standard KD loss function combines the standard cross-entropy loss with a distillation loss term:

<!-- $$\mathbb{P}[M(D)\in A]\leq e^{\alpha} \cdot \mathbb{P}[M(D')\in A]$$ -->

$$\mathcal{L}_{KD}=(1-\alpha)\mathrm{CE}(y,\sigma(z_s))+\alpha T^2 \mathrm{CE}(\sigma(z_t^T),\sigma(z_s^T))$$

Where:
- $z_s$ and $z_t$ are the logits from the student and teacher models
- $T$ is the temperature parameter that controls softening of probability distributions
- $z_s^T \mathrel{:}= \frac{z_s}{T}$ and $z_t^T \mathrel{:}= \frac{z_t}{T}$
- $\sigma$ is the softmax function
- $\sigma(z_s^T) \mathrel{:}= \frac{\exp(z_s^T)}{\sum_j \exp(z_j^T)}$ and $\sigma(z_t^T) \mathrel{:}= \frac{\exp(z_t^T)}{\sum_j \exp(z_j^T)}$
- $\mathrm{CE}$ is cross-entropy loss
- $\alpha$ balances the importance of each loss component.


The first part of the loss $(1-\alpha)\mathrm{CE}(y,\sigma(z_s))$ is to incitate the student model to learn from one hot encoded ground truth label.

The second part of the loss $\alpha T^2 \mathrm{CE}(\sigma(z_t^T),\sigma(z_s^T))$ is to incitate the student model to try to reproduce the ouputs of the teacher model. This is what permits the student to learn from the teacher.
The larger $\alpha$ is, the more the student will try to replicate the teacher model's outputs and ignore the one hot encoded groundtruth and vice versa.


### Label Smoothing
Label smoothing (LS) is another technique that smooths hard targets by mixing them with a uniform distribution. In the cross entropy loss we replace the one hot encoded $y$ by $y_{LS} \mathrel{:}= (1-\alpha)y + \frac{\alpha}{K}$, where $K$ is the number of classes and $\alpha$ the smoothing parameter:

<script type="math/tex; mode=display">
\begin{align}
CE(y_{LS},\sigma(z)) &= - \sum_{i=1}^{K} \left( (1 - \alpha) y_i + \frac{\alpha}{K} \right) \log \sigma(z_i) \\
&= -(1 - \alpha) \sum_{i=1}^{K} y_i \log \sigma(z_i) - \alpha \sum_{i=1}^{K} \frac{1}{K}\log \sigma(z_i) \\
\end{align}
</script>

We obtain a loss that is similar to knowledge diffusion but there is a key difference important for interpretability that we will discuss later.
From the equation above, we get the label smoothing loss equation:
$$L_{LS} = (1-\alpha)\mathrm{CE}(y,\sigma(z)) + \alpha\mathrm{CE}(u,\sigma(z)) $$
Where $u$ is a uniform distribution over all the possible $K$ classes.

<!--
### Knowledge Distillation vs. Label Smoothing

Label smoothing (LS) is another technique that smooths hard targets by mixing them with a uniform distribution. While it might seem similar to KD, the paper shows fundamental differences:

- **Label Smoothing**: $y_{LS} = (1-\epsilon)y + \epsilon/K$, where $K$ is the number of classes and $\epsilon$ is a small constant
- **Knowledge Distillation**: Uses teacher's soft predictions that contain rich class similarity information

This distinction is crucial for interpretability, as we'll see later.
-->

<!--
### Basic Knowledge Distillation Implementation
Y'a peut etre une erreur dans le code ici dquand on divisve par la temperature je pense que la division de s'applique pas dans le dénominateur du softmax mais je ne suis pas sur
Let's look at a basic PyTorch implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def knowledge_distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=4.0):
    """
    Compute the knowledge distillation loss.

    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        targets: Ground truth labels
        alpha: Weight for distillation loss vs. standard cross-entropy loss
        temperature: Temperature for softening probability distributions

    Returns:
        Total loss combining cross-entropy and distillation loss
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, targets)

    # Soften logits with temperature
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    # KL divergence loss
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Combine losses
    total_loss = (1 - alpha) * ce_loss + alpha * kd_loss

    return total_loss
```
-->

<h2 id="ii-defining-interpretability-through-network-dissection" style="font-size: 21px; display: flex; align-items: center;"> II. Defining Interpretability Through Network Dissection </h2>

The first thing to know is that there are different approaches to define and measure interpretability in machine learning.

For image classification, the authors use <a href="https://arxiv.org/pdf/1711.05611v2">network dissection</a> to quantitatively measure interpretability. The idea is to compare activation maps and see if areas with high activation correspond to an object or a meaningful concept on the image.

The process can be better understood through the following illustration:

<p align="center">
  <img src="/images/Bryan_Remi/network_dissection.png" alt="Network Dissection Process" width="700">
</p>



Feed a neural network model an image, pick a deep layer and count the number of neurons that detects a concept like "cat" or "dog".
We call those neurons concept detectors and will define them more precisely. The number of concept detectors will be the primary metric to define the interpretability of a model, the higher the more we will consider it interpretable.



**The easiest way to understand what is a concept detector is to look at the following pseudo code to compute the number of concept detectors:**


1. First, we need to choose a layer $\mathcal{l}$ to **dissect**, typically deep in the network.
2. For each image x in the dataset,

   2.a) Feed an image **x** of shape (n,n) into the neural network.

   2.b) For each neuron in the layer $\mathcal{l}$, we collect the activation maps
   **A<sub>i</sub>(x)** ∈ ℝ<sup>d×d</sup>, where **d < n** and i is the index of the neuron.

3. For each neuron in the layer **l**, we define **a<sub>i</sub>** as the distribution of
   individual unit activation. This can be thought of as an empirical distribution where
   the varying element is the image **x**. Each image contributes to construct **a<sub>i</sub>**.

4. For each neuron, we compute a threshold **T<sub>i</sub>** such that
   P(a<sub>i</sub> ≥ T<sub>i</sub>) = 0.005. That will be usefull to only keep top 0.5% of activations.

5. We interpolate **A<sub>i</sub>** to match the dimension of the image of shape (n,n)
   to enable comparisons.

6. For each image x in the dataset,

   6.a) We create a binary mask **A<sub>i</sub><sup>mask</sup>(x)** of shape (n,n),
   where each element is 1 if its corresponding element in **A<sub>i</sub>(x)**
   is greater than or equal to the threshold **T<sub>i</sub>**.
   This allows us to consider only the top 0.5% activations.

   6.b) We are provided with another mask **M<sub>c</sub>(x)** of shape (n,n),
    where each element is 1 if its corresponding pixel in the image **x**
    is labeled with the ground truth class **c**, and 0 otherwise.

   6.c) We compute the intersection over union **IoU<sub>i,c</sub>** between
    **A<sub>i</sub><sup>mask</sup>(x)** and **M<sub>c</sub>(x)**. If the intersection over union **IoU<sub>i,c</sub>** is larger than a fixed threshold (0.05),
    then neuron **i** is considered a **concept detector** for concept **c**.






### If you prefer to understand with code, here is an implementation of the procedure described above:

```python
def identify_concept_detectors(model, layer_name, dataset, concept_masks):
    """
    Identify neurons that act as concept detectors in a specific layer.

    Args:
        model: Neural network model
        layer_name: Name of the layer to analyze
        dataset: Dataset with images
        concept_masks: Dictionary mapping images to concept segmentation masks

    Returns:
        Dictionary mapping neurons to detected concepts
    """
    # Step 1: Collect activation maps for each image
    activation_maps = {}

    for image in dataset:
        # Forward pass and extract activation at specified layer
        activations = get_layer_activation(model, layer_name, image)

        for neuron_idx, activation in enumerate(activations):
            if neuron_idx not in activation_maps:
                activation_maps[neuron_idx] = []
            activation_maps[neuron_idx].append(activation)

    # Step 2: Compute threshold for top 0.5% activations for each neuron
    thresholds = {}
    for neuron_idx, activations in activation_maps.items():
        # Flatten all activations for this neuron
        all_activations = torch.cat([act.flatten() for act in activations])
        # Compute threshold for top 0.5%
        threshold = torch.quantile(all_activations, 0.995)
        thresholds[neuron_idx] = threshold

    # Step 3: Create binary masks and compute IoU with concept masks
    concept_detectors = {}

    for image_idx, image in enumerate(dataset):
        image_concepts = concept_masks[image_idx]

        for neuron_idx, activations in activation_maps.items():
            # Get activation for this neuron on this image
            activation = activations[image_idx]

            # Create binary mask using threshold
            binary_mask = (activation >= thresholds[neuron_idx]).float()

            # Resize to match image dimensions
            binary_mask = F.interpolate(
                binary_mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[1:],
                mode='bilinear'
            ).squeeze()

            # Compute IoU with each concept mask
            for concept, mask in image_concepts.items():
                intersection = torch.sum(binary_mask * mask)
                union = torch.sum(binary_mask) + torch.sum(mask) - intersection
                iou = intersection / union if union > 0 else 0

                # If IoU exceeds threshold (typically 0.05), consider it a detector
                if iou > 0.05:
                    if neuron_idx not in concept_detectors:
                        concept_detectors[neuron_idx] = set()
                    concept_detectors[neuron_idx].add(concept)

    return concept_detectors
```


<h2 id="iii-why-knowledge-distillation-enhances-interpretability" style="font-size: 21px; display: flex; align-items: center;"> III. Why Knowledge Distillation Enhances Interpretability </h2>

The key insight from the paper is that knowledge distillation transfers not just the ability to classify correctly, but also class similarity information that makes the model focus on more interpretable features.

### Transfer of Class Similarities

When a teacher model sees an image of a dog, it might assign:
- 85% probability to "Golden Retriever"
- 10% probability to other dog breeds
- 5% probability to other animals and objects

These "soft targets" encode rich hierarchical information about how classes relate. The student model distilling this knowledge learns to focus on features that are common to similar classes (e.g., general "dog" features).

### Label Smoothing vs. Knowledge Distillation
By looking at the KD and label smoothing losses, we can see that they are similar. When $T=1$ they only differ in the second member where we have a $\sigma(z_t^T)$ that contains class similarity information instead of $u$ that doesn't contain any information.



 * $\mathcal{L}_{KD}=(1-\alpha)\mathrm{CE}(y,\sigma(z_s))+\alpha T^2 \mathrm{CE}(\sigma(z_t^T),\sigma(z_s^T))$
 * $L_{LS} = (1-\alpha)\mathrm{CE}(y,\sigma(z)) + \alpha\mathrm{CE}(u,\sigma(z)) $

 So, if there is a difference in interpretability, it is likely that it comes from the fact that distillation permits to get class similarity knowledge from the teacher model. This is exactly what is shown in the figure below. Knowledge distillation guides student models to focus on more object-centric features rather than background or contextual features. This results in activation maps that better align with the actual objects in images.




<!-- <p align="center">
  <img src="/images/Bryan_Remi/activation_comparison.png" alt="Activation Map Comparison" width="700">
</p> -->

<p align="center">
  <img src="/images/Bryan_Remi/comparisons_dog.png" alt="ObjectCentricActivation" width="700">
</p>






The next figure also highlights the loss of interpretability (less concept detectors) when using label smoothing and the improvement of interpretability (more concept detectors) for KD:

<p align="center">
  <img src="/images/Bryan_Remi/NbConceptDetDiffModels.png" alt="KD vs LS Distributions" width="600">
</p>
<!--
- **Label Smoothing**: Replaces one-hot with a uniform mixture, removing class similarity information
- **Knowledge Distillation**: Transfers structured class similarities from teacher to student
-->
While label smoothing can improve accuracy, it often reduces interpretability by erasing valuable class relationships while KD keep class relationship information and improves both accuracy and interpretability.


<h2 id="iv-experimental-results-and-reproduction" style="font-size: 21px; display: flex; align-items: center;"> IV. Experimental Results and Reproduction </h2>

Let's implement a reproduction of one of the paper's key experiments to see knowledge distillation's effect on interpretability in action.

### Setting Up the Experiment

We are going to replicate the experiment by using the <a href="https://github.com/Rok07/KD_XAI">GitHub repository provided by the authors</a>. The repository contains the code to train the models, compute the concept detectors, and evaluate the interpretability of the models.

As it is often the case with a machine learning paper, running the code to reproduce results requires some struggle.
To reproduce the results, you could use a virtual environment and then do the following:

```bash
git clone https://github.com/Rok07/KD_XAI.git
mkdir /tmp/
mv KD_XAI /tmp/
cd /tmp/KD_XAI/
cd torchdistill
pip install -e .
cd ..
bash script/dlbroden.sh
nano torchdistill/torchdistill/models/custom/bottleneck/__init__.py
~ comment the first line
pip install opencv-python
pip install imageio
sudo apt update
sudo apt install -y libgl1-mesa-glx
nano /tmp/KD_XAI/util/vecquantile.py
~ change NaN by nan
nano /tmp/KD_XAI/loader/data_loader.py
~ add out[i] = rgb[:,:,0] + (rgb[:,:,1].astype(np.uint16) * 256)
nano /tmp/KD_XAI/settings.py
~ change BATCH_SIZE=16
~ change WORKERS=4
python main.py
```
<h2 id="v-beyond-network-dissection-other-interpretability-metrics" style="font-size: 21px; display: flex; align-items: center;"> V. Beyond Network Dissection: Other Interpretability Metrics </h2>

<p>While the paper emphasizes the use of <strong>Network Dissection</strong> to measure model interpretability by quantifying concept detectors, it also explores several additional metrics to confirm the broader impact of <strong>Knowledge Distillation (KD)</strong> on interpretability:</p>

<ul>
  <li><strong>Five-Band Scores:</strong> This metric assesses interpretability by evaluating pixel accuracy (accuracy of saliency maps in identifying critical features), precision (how well the saliency maps match the actual distinguishing features), recall, and false positive rates (FPR, lower FPR indicates better interpretability) using a synthesized dataset with heatmap ground truths. KD-trained models consistently show higher accuracy and lower FPR compared to other methods.</li>
  <li><strong><a href="https://arxiv.org/pdf/2102.12781">DiffROAR Scores</a>, proposed by Shah et al. (2021):</strong> This evaluates the difference in predictive power on a model trained on a dataset and a model trained on a version of the dataset where we removed top and bottom x% of the pixel according to their importance for the task. The authors find that KD has a higher DiffROAR score than a model trained from scratch. It means that KD makes the model use more relevant features and thus more interpretable in that sense.


<!--  This evaluates the degree to which instance-specific explanations focus on discriminative features. KD-trained models outperform others in aligning these explanations effectively.</li> -->


  <li><strong>Loss Gradient Alignment:</strong> This metric measures the alignment of model gradients with human-perceived important features. KD models exhibit better alignment, indicating greater interpretability as we can see on this figure:
  <p align="center">
  <img src="/images/Bryan_Remi/gradient_interpre.png" alt="ObjectCentricActivation" width="700">
</p>


  </li>
</ul>


<p>
<p>These metrics collectively show that KD can enhance interpretability. The consistent results showing that knowledge distillation can enhance interpretability for different metrics of interpretability provide strong arguments to believe that KD could be broadly used for better interpretability of machine learning models. </p>

<!--
<p>These metrics collectively demonstrate that KD enhances interpretability across diverse datasets and domains, extending beyond the visual tasks traditionally associated with these evaluations. The inclusion of these metrics highlights KD's consistent advantages beyond traditional measures like concept detectors, providing a more holistic perspective on interpretability.</p> -->


<h2 id="conclusion" style="font-size: 21px; display: flex; align-items: center;"> Conclusion </h2>

<p align="center"> <img src="/images/Bryan_Remi/pinguins_studying.gif" alt="Feeling strong with interpretable AI" style="width: 30%; max-width: 500px; height: auto;"> </p>


The article showed that knowledge distillation can improve both accuracy and interpretability. They attribute the improvement in interpretability to the transfer of class similarity knowledge from the teacher to the student model. They compare label smoothing (LS) that is similar to KD but LS does not benefit from class similarity information. The empirical experiments shows better accuracy for LS and KD but the interpretability of LS descreases whereas it increases for KD confirming the hypothesis that class similarity knowledge has a role in interpretability. The authors obtain consistent results when using other metrics than the number of concept detectors for interpretability showing that their approach is robust to different definitions of interpretability.

Those encouraging results could lead to applications of knowledge distillation to improve the interpretability of machine learning models in highly sensitive areas like autonomous systems and healthcare.

<!--The research sheds light on a game-changing discovery: knowledge distillation not only boosts performance but significantly enhances model interpretability. By transferring class-similarity information, KD ensures that student models become more interpretable. From applications in healthcare to autonomous systems, this insight could redefine how we approach AI development. What’s your take on making AI models more transparent and interpretable? -->






<h2 id="join-the-discussion" style="font-size: 21px; display: flex; align-items: center;"> Join the Discussion </h2>

We’d love to hear your thoughts! What are your experiences with Knowledge Distillation (KD)? Have you found it to improve not just performance but also interpretability in your projects? Feel free to share your ideas, questions, or insights in the comments section or engage with us on <a href="https://github.com/BryanBradfo/responsible-ai-datascience-ipParis.github.io">GitHub</a>!








<!--

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

-->

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). <a href="https://arxiv.org/abs/1503.02531">Distilling the knowledge in a neural network.</a> arXiv:1503.02531.
- Han, H., Kim, S., Choi, H.-S., & Yoon, S. (2023). <a href="https://arxiv.org/pdf/2305.15734">On the Impact of Knowledge Distillation for Model Interpretability.</a> arXiv:2305.15734.
- Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). <a href="https://arxiv.org/pdf/1704.05796">Network dissection: Quantifying interpretability of deep visual representations.</a> arXiv:1704.05796.
- Shah, H., Jain, P., & Netrapalli, P. (2021). <a href="https://arxiv.org/pdf/2102.12781">Do input gradients highlight discriminative features?</a> arXiv:2102.12781, NeurIPS 2021.


</html>
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
         inlineMath: [['$','$'], ['\$','\$']],
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

Knowledge distillation is a powerful technique to transfer the knowledge from a large "teacher" model to a "student" model. While it's commonly used to improve performance and reduce computational costs by compressing large models, this blog post explores a fascinating discovery: knowledge distillation can also enhance model interpretability. We'll dive into the paper <a href="https://arxiv.org/abs/2305.15734">On the Impact of Knowledge Distillation for Model Interpretability"</a> (ICML 2023) by H. Han et al., which sheds light on this novel perspective.

<p align="center">
  <img src="/images/Bryan_Remi/better_meme.png" alt="Introduction" style="width: 60%; max-width: 500px; height: auto;">
</p>

Interpretability in AI allows researchers, engineers, and decision-makers to trust and control machine learning models. Recent models show impressive performance on many different tasks and often rely on deep learning models. Unfortunately, deep learning models are also know for the difficulty to interprete them and understand how they come to a result wich can be problematic in highly sensitive applications like autonomous driving or healthcare. The article we present in this blog shows that knowledge distillation can improve the interpretability of deep learning models.


When AI is a black box, you're just hoping for the best. But when you understand it, you become unstoppable.

<h2 style="font-size: 21px; display: flex; align-items: center;"> 0. Table of Contents </h2>

- [I. Crash Course on Knowledge Distillation and Label Smoothing](#i-crash-course-on-knowledge-distillation)
- [II. Defining Interpretability Through Network Dissection](#ii-defining-interpretability-through-network-dissection)
- [III. Logit Distillation & Feature Distillation: A Powerful Duo for Interpretability](#iii-logit-distillation-feature-distillation)
- [IV. Why Knowledge Distillation Enhances Interpretability](#iv-why-knowledge-distillation-enhances-interpretability)
- [V. Experimental Results and Reproduction](#v-experimental-results-and-reproduction)
- [VI. Beyond Network Dissection: Other Interpretability Metrics](#vi-beyond-network-dissection-other-interpretability-metrics)
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

<p align="center">
  <img src="/images/Bryan_Remi/label_smoothing.jpg" alt="Label Smoothing" width="700">
</p>


<h2 id="ii-defining-interpretability-through-network-dissection" style="font-size: 21px; display: flex; align-items: center;"> II. Defining Interpretability Through Network Dissection </h2>

The first thing to know is that there are different approaches to define and measure interpretability in machine learning.

For image classification, the authors use <a href="https://arxiv.org/pdf/1711.05611v2">network dissection</a> to quantitatively measure interpretability. The idea is to compare activation maps and see if areas with high activation correspond to an object or a meaningful concept on the image.

The process can be better understood through the following illustration:

<p align="center">
  <img src="/images/Bryan_Remi/network_dissection.png" alt="Network Dissection Process" width="700">
</p>



Feed a neural network model an image, pick a deep layer and count the number of neurons that detects a concept like "cat" or "dog".
We call those neurons concept detectors and will define them more precisely. The **number of concept detectors will be the primary metric to define the interpretability of a model**, the higher the more we will consider it interpretable.



**The easiest way to understand what is a concept detector is to look at the following pseudo code to compute the number of concept detectors:**

<style>
  body {
      font-family: 'Arial', sans-serif;
      line-height: 1.6;
  }

  .steps-container {
      background: #f8f9fa;
      border-left: 5px solid #007bff;
      padding: 15px 20px;
      margin: 20px 0;
      border-radius: 5px;
  }

  .step {
      font-weight: bold;
      color: #007bff;
      margin-top: 15px;
  }

  .math-expression {
      font-family: 'Courier New', Courier, monospace;
      background: #e9ecef;
      padding: 5px;
      border-radius: 3px;
  }

  .important {
      background: #fff3cd;
      color: #856404;
      padding: 10px;
      border-left: 4px solid #ffc107;
      border-radius: 3px;
      margin: 10px 0;
  }
</style>

<div class="steps-container">

### <span class="step">1. Selecting the Layer</span>
First, we need to choose a layer $\mathcal{l}$ to **dissect**, typically deep in the network.

### <span class="step">2. Processing Each Image</span>
For each image **x** in the dataset:

1. **Feedforward Pass**: 
   - Input an image **x** of shape $ (n,n) $ into the neural network.

2. **Activation Extraction**:
   - For each neuron in layer $\mathcal{l}$, collect the activation maps:
     <div class="math-expression">
     \[ A_i(x) \in \mathbb{R}^{d \times d}, \quad \text{where } d < n \text{ and } i \text{ is the neuron index.} \]
     </div>

### <span class="step">3. Defining Activation Distribution</span>
For each neuron **i** in the layer $\mathcal{l}$:

- Define **a<sub>i</sub>** as the empirical distribution of activation values across different images **x**.

### <span class="step">4. Computing Activation Threshold</span>
- Compute a threshold **T<sub>i</sub>** such that:
  <div class="math-expression">
  \[ P(a_i \geq T_i) = 0.005 \]
  </div>
  - This ensures only the **top 0.5%** activations are retained.

### <span class="step">5. Resizing Activation Maps</span>
- Interpolate **A<sub>i</sub>** to match the dimension $ (n,n) $ for direct comparison with input images.

### <span class="step">6. Creating Binary Masks</span>
For each image **x**:

1. **Generating Activation Masks**:
   - Create a **binary mask** $ A_i^{\text{mask}}(x) $ of shape $ (n,n) $:
     <div class="math-expression">
     \[ A_i^{\text{mask}}(x)[j,k] = \begin{cases} 1, & \text{if } A_i(x)[j,k] \geq T_i \\ 0, & \text{otherwise} \end{cases} \]
     </div>
   - This retains only the highest activations.

2. **Using Ground Truth Masks**:
   - Given a **ground truth mask** $ M_c(x) $ of shape $ (n,n) $, where:
     - $ M_c(x)[j,k] = 1 $ if the pixel in **x** belongs to class **c**, otherwise **0**.

3. **Computing Intersection over Union (IoU)**:
   - Calculate the IoU between **A<sub>i</sub><sup>mask</sup>(x)** and **M<sub>c</sub>(x)**:
     <div class="math-expression">
     \[ \text{IoU}_{i,c} = \frac{|A_i^{\text{mask}}(x) \cap M_c(x)|}{|A_i^{\text{mask}}(x) \cup M_c(x)|} \]
     </div>
   - If  $\text{IoU}_{i,c} > 0.05$, the neuron **i** is considered a **concept detector** for concept **c**.

</div>



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

<h2 id="iii-logit-distillation-feature-distillation" style="font-size: 21px; display: flex; align-items: center;"> III. Logit Distillation & Feature Distillation: A Powerful Duo for Interpretability </h2>

Combining logit distillation with feature distillation not only boosts performance but also enhances the interpretability of student models. This improvement is measured by an increase in the number of concept detectors, which represent units aligned with human-interpretable concepts.

<p align="center">
  <img src="/images/Bryan_Remi/feature_logit_distillation.png" alt="Feature_Logit_Distillation" width="700">
</p>

where Attention Transfer (AT), Factor Transfer (FT), Contrastive Representation Distillation (CRD), and Self-Supervised Knowledge Distillation (SSKD) are all variations of knowledge distillation techniques, each designed to transfer knowledge from teacher models to student models in unique ways.

### How they work together?

1. **Logit Distillation:**

- Transfers class-similarity information from the teacher to the student through softened logits.
- Helps the student model understand the relationships between semantically similar classes, making activation maps more object-centric.

2. **Feature Distillation:**

- Focuses on aligning intermediate layer features between the teacher and student.
- Improves the student model's ability to replicate the teacher’s feature representations, supporting richer internal representations.


<h2 id="iv-why-knowledge-distillation-enhances-interpretability" style="font-size: 21px; display: flex; align-items: center;"> IV. Why Knowledge Distillation Enhances Interpretability </h2>

The key insight from the paper is that knowledge distillation transfers not just the ability to classify correctly, but also class-similarity information that makes the model focus on more interpretable features.

### Transfer of Class Similarities

When a teacher model sees an image of a dog, it might assign:
- 85% probability to "Golden Retriever"
- 10% probability to other dog breeds
- 5% probability to other animals and objects

These "soft targets" (consequence of logit distillation) encode rich hierarchical information about how classes relate. The student model distilling this knowledge learns to focus on features that are common to similar classes (e.g., general "dog" features).


### Label Smoothing vs. Knowledge Distillation

By looking at the KD and label smoothing losses, we can see that they are similar. When $T=1$ they only differ in the second member where we have a $\sigma(z_t^T)$ that contains class-similarity information instead of $u$ that doesn't contain any information.

* $\mathcal{L}_{KD}=(1-\alpha)\mathrm{CE}(y,\sigma(z_s))+\alpha T^2 \mathrm{CE}(\sigma(z_t^T),\sigma(z_s^T))$
* $L_{LS} = (1-\alpha)\mathrm{CE}(y,\sigma(z)) + \alpha\mathrm{CE}(u,\sigma(z)) $

So, if there is a difference in interpretability, it is likely that it comes from the fact that distillation permits to get class similarity knowledge from the teacher model. This is exactly what is shown in the figure below. Knowledge distillation guides student models to focus on more object-centric features rather than background or contextual features. This results in activation maps that better align with the actual objects in images.

<p align="center">
  <img src="/images/Bryan_Remi/comparisons_dog.png" alt="ObjectCentricActivation" width="700">
</p>

The next figure also highlights the loss of interpretability (less concept detectors) when using label smoothing and the improvement of interpretability (more concept detectors) for KD:

<p align="center">
  <img src="/images/Bryan_Remi/NbConceptDetDiffModels.png" alt="KD vs LS Distributions" width="600">
</p>

While label smoothing can improve accuracy, it often reduces interpretability by erasing valuable class relationships while KD keeps class relationship information and improves both accuracy and interpretability.


<h2 id="v-experimental-results-and-reproduction" style="font-size: 21px; display: flex; align-items: center;"> V. Experimental Results and Reproduction </h2>

Let's implement a reproduction of one of the paper's key experiments to see knowledge distillation's effect on interpretability in action.

### Setting Up the Experiment

We are going to replicate the experiment by using the <a href="https://github.com/Rok07/KD_XAI">GitHub repository provided by the authors</a>. The repository contains the code to train the models, compute the concept detectors, and evaluate the interpretability of the models.

As it is often the case with a machine learning paper, running the code to reproduce results requires some struggle.
To reproduce the results, you could use a virtual environment (e.g. <a href="https://datalab.sspcloud.fr/">SSP Cloud Datalab</a>) and then do the following:

```bash
git clone https://github.com/Rok07/KD_XAI.git
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
nano util/vecquantile.py
~ change NaN by nan
nano loader/data_loader.py
~ add out[i] = rgb[:,:,0] + (rgb[:,:,1].astype(np.uint16) * 256)
cd ..
nano settings.py
~ change TEST_MODE = False to True
cd dataset/broden1_224
cp index.csv index_sm.csv
~ keep the 4000 first lines
cd ../..
nano visualize/bargraph.py
~ change parameter threshold of bar_graph_svg() to 0.001
python main.py
```

Network Dissection quantifies the interpretability of hidden units by measuring their alignment with human-interpretable concepts. The following results reveal several interesting findings:

#### 1. Concept Distribution (from bargraph.svg):

<p align="center">
  <img src="/images/Bryan_Remi/class_distribution.png" alt="Class Distribution" width="600">
</p>

   - ~6 units detecting object concepts
   - ~2 units detecting scene concepts
   - 1 unit detecting material properties
   - ~13 units detecting textures
   - ~6 units detecting colors


#### 2. Specific Units: (layer4-0xxx.jpg)

<p align="center">
  <img src="/images/Bryan_Remi/unit_grid.png" alt="Unit Grid" width="1600">
</p>

   - **Unit 330** has specialized in detecting grid and regular pattern textures

<p align="center">
  <img src="/images/Bryan_Remi/unit_sky.png" alt="Unit Sky" width="1600">
</p>

   - **Unit 202** detects sky regions in images

The network dissection approach reveals interpretable neurons of a distilled ResNet18.

<h2 id="vi-beyond-network-dissection-other-interpretability-metrics" style="font-size: 21px; display: flex; align-items: center;"> VI. Beyond Network Dissection: Other Interpretability Metrics </h2>

<p>While the paper emphasizes the use of <strong>Network Dissection</strong> to measure model interpretability by quantifying concept detectors, it also explores several additional metrics to confirm the broader impact of <strong>Knowledge Distillation (KD)</strong> on interpretability:</p>

<ul>
  <li><strong><a href="https://arxiv.org/pdf/2009.02899">Five-Band Scores</a>, proposed by Tjoah & Guan (2020):</strong> This metric assesses interpretability by evaluating pixel accuracy (accuracy of saliency maps in identifying critical features), precision (how well the saliency maps match the actual distinguishing features), recall, and false positive rates (FPR, lower FPR indicates better interpretability) using a synthesized dataset with heatmap ground truths. KD-trained models consistently show higher accuracy and lower FPR compared to other methods.</li>
  <li><strong><a href="https://arxiv.org/pdf/2102.12781">DiffROAR Scores</a>, proposed by Shah et al. (2021):</strong> This evaluates the difference in predictive power on a model trained on a dataset and a model trained on a version of the dataset where we removed top and bottom x% of the pixel according to their importance for the task. The authors find that KD has a higher DiffROAR score than a model trained from scratch. It means that KD makes the model use more relevant features and thus more interpretable in that sense.

  <li><strong>Loss Gradient Alignment:</strong> This metric measures the alignment of model gradients with human-perceived important features. KD models exhibit better alignment, indicating greater interpretability as we can see on this figure:
  <p align="center">
  <img src="/images/Bryan_Remi/gradient_interpre.png" alt="ObjectCentricActivation" width="700">
</p>


  </li>
</ul>


<p>
<p>These metrics collectively show that KD can enhance interpretability. The consistent results showing that knowledge distillation can enhance interpretability for different metrics of interpretability provide strong arguments to believe that KD could be broadly used for better interpretability of deep learning models. </p>



<h2 id="conclusion" style="font-size: 21px; display: flex; align-items: center;"> Conclusion </h2>

<p align="center"> <img src="/images/Bryan_Remi/pinguins_studying.gif" alt="Feeling strong with interpretable AI" style="width: 30%; max-width: 500px; height: auto;"> </p>


The article showed that knowledge distillation can improve both accuracy and interpretability. They attribute the improvement in interpretability to the transfer of class similarity knowledge from the teacher to the student model. They compare label smoothing (LS) that is similar to KD but LS does not benefit from class-similarity information. The empirical experiments shows better accuracy for LS and KD but the interpretability of LS decreases whereas it increases for KD confirming the hypothesis that class similarity knowledge has a role in interpretability. The authors obtain consistent results when using other metrics than the number of concept detectors for interpretability showing that their approach is robust to different definitions of interpretability.

Those encouraging results could lead to applications of knowledge distillation to improve the interpretability of deep learning models in highly sensitive areas like autonomous systems and healthcare.



<h2 id="join-the-discussion" style="font-size: 21px; display: flex; align-items: center;"> Join the Discussion </h2>

We’d love to hear your thoughts! What are your experiences with Knowledge Distillation (KD)? Have you found it to improve not just performance but also interpretability in your projects? Feel free to share your ideas, questions, or insights in the comments section or engage with us on <a href="https://github.com/BryanBradfo/responsible-ai-datascience-ipParis.github.io">GitHub</a>!




## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). <a href="https://arxiv.org/abs/1503.02531">Distilling the knowledge in a neural network.</a> arXiv:1503.02531.
- Han, H., Kim, S., Choi, H.-S., & Yoon, S. (2023). <a href="https://arxiv.org/pdf/2305.15734">On the Impact of Knowledge Distillation for Model Interpretability.</a> arXiv:2305.15734.
- Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). <a href="https://arxiv.org/pdf/1704.05796">Network dissection: Quantifying interpretability of deep visual representations.</a> arXiv:1704.05796.
- Tjoa, E., & Guan, M. Y. (2020). <a href="https://arxiv.org/pdf/2009.02899"> Quantifying explainability of saliency methods in deep neural networks.</a> arXiv:2009.02899.
- Shah, H., Jain, P., & Netrapalli, P. (2021). <a href="https://arxiv.org/pdf/2102.12781">Do input gradients highlight discriminative features?</a> arXiv:2102.12781, NeurIPS 2021.


</html>
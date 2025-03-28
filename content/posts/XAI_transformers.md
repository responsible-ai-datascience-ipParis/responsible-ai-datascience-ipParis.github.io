+++
title = 'XAI_transformers'
date = 2025-03-28T16:37:43+01:00
draft = false
+++

<!DOCTYPE html>
<html lang="fr">
<head>
   <meta charset="UTF-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <title>XAI for Transformers</title>
   <style>
      body {
         font-family: Arial, sans-serif;
         line-height: 1.6;
         margin: 40px;
         background-color: #fff;
         color: #111;
      }
      h1 {
         font-size: 28px;
         text-align: center;
         margin-bottom: 5px;
      }
      .author-section {
         display: flex;
         gap: 20px;
         margin-top: 20px;
         margin-bottom: 20px;
      }
      .author-box {
         display: flex;
         align-items: center;
      }
      .author-box img {
         width: 40px;
         height: 40px;
         border-radius: 50%;
         margin-right: 10px;
      }
      .toc {
         background-color: #f0f0f0;
         padding: 15px;
         border-radius: 8px;
         margin-top: 30px;
         margin-bottom: 30px;
      }
      .toc ul {
         padding-left: 20px;
         margin: 0;
      }
      .toc li {
         margin-bottom: 6px;
      }
      blockquote {
         font-style: italic;
         color: #555;
         border-left: 4px solid #ccc;
         padding-left: 15px;
         margin: 20px 0;
      }
      img.centered {
         display: block;
         margin: 20px auto;
         max-width: 60%;
      }
      h3, h4 {
         margin-top: 30px;
      }
   </style>
</head>

<body>

<h1>XAI for Transformers: Better explanations through conservative propagation</h1>
<p style="text-align:center;">Published March 28, 2025</p>

<div class="author-section">
   <div class="author-box">
      <img src="/images/illustrations_XAI/audrey.png" alt="Audrey Airaud">
      <div>
         <a href="https://github.com/odd2022">odd2022</a><br>
         <span>Audrey Airaud</span>
      </div>
   </div>
   <div class="author-box">
      <img src="/images/illustrations_XAI/foucauld.png" alt="Foucauld Estignard">
      <div>
         <a href="https://github.com/FoucauldE">FoucauldE</a><br>
         <span>Foucauld Estignard</span>
      </div>
   </div>
   <div class="author-box">
      <img src="/images/illustrations_XAI/nathan.png" alt="Nathan Toumi">
      <div>
         <a href="https://github.com/N22Toumi">N22Toumi</a><br>
         <span>Nathan Toumi</span>
      </div>
   </div>
</div>

<p>This is a blog post about the article “XAI for Transformers: Better Explanations through Conservative Propagation” published by Ameen Ali et al. in 2022 and available <a href="https://arxiv.org/pdf/2202.07304"><strong>here</strong></a>.</p>

<blockquote>
Sure, here is a new version of your article on XAI for Transformers with a more adapted tone:
</blockquote>

<p>You’ve probably encountered similar sentences before — they are characteristic of recent language models. These models are built on a specific architecture known as the Transformer, which has extended far beyond Natural Language Processing (NLP) to fields like computer vision, graph analysis, and audio signal processing. 

While models relying on Transformers have shown impressive performance, their behavior remains hard to explain, raising questions about their use in sensitive domains like healthcare <a href="#health1">[1]</a><a href="#health2">[2]</a>, cybersecurity <a href="#cyber1">[3]</a><a href="#cyber2">[4]</a>, recruitment <a href="#recr">[5]</a> or education <a href="#edu">[6]</a>. Understanding their decisions therefore becomes a major challenge, to ensure that they do not discriminate on unwanted features (eg. gender, ethnicity). </p>

<p align="center">
  <img src="/images/illustrations_XAI/illustration_intro.png" alt="illustration_intro">
</p>

<!-- TABLE OF CONTENTS -->
<div class="toc">
   <strong>Table of Contents</strong>
                    <li><a href="#apply-transformers">3.2 Apply directly to transformer architectures?</a></li>
         </ul>
      </li>
      <li><a href="#fixing-breaks">4. Fixing conservation breaks: a simple but effective trick</a>
         <ul>
            <li><a href="#Attention-Heads">4.1 Locally linear expansion for Attention Heads</a></li>
            <li><a href="#LayerNorm">4.2 Locally linear expansion for LayerNorm</a></li>
            <li><a href="#implementation">4.3 Implementation made easy</a></li>
         </ul>
      </li>
      <li><a href="#experiments">5. Confirmation with Experiments</a>
         <ul>
            <li><a href="#quantitative-results">5.1 Quantitative Results</a></li>
            <li><a href="#qualitative-results">5.2 Qualitative Comparison</a></li>
         </ul>
      </li>
      <li><a href="#conclusion">6. Key Takeaway</a></li>
      <li><a href="#references">References</a></li>
   </ul>
</div>

<!-- ARTICLE CONTINUES AS-IS -->

<h3 id="attribution-methods">1. Attribution methods: how to identify important features?</h3>
<p>
In order to make deep learning models more interpretable, especially in critical applications, it is crucial to understand which input features contribute the most to a model’s prediction. This has given rise to a class of techniques known as <strong>attribution methods</strong>, whose goal is to assign relevance scores to input features based on their influence on the model’s output.
</p>

<p>
Formally, consider a function $F: \mathbb{R}^n \rightarrow [0, 1]$ representing a deep network, and an input $x = (x_1, \dots, x_n) \in \mathbb{R}^n$. An <strong>attribution</strong> of the prediction at input $x$ relative to a baseline input $x'$ is a vector: 
$A_F(x, x') = (a_1, \dots, a_n) \in \mathbb{R}^n$,
where each $a_i$ represents the contribution of feature $x_i$ to the prediction $F(x)$.
</p>

<p>
Numerous attribution techniques have been proposed, each relying on different strategies to assess feature importance. These methods generally fall into three categories <a href="#integrated-gradients">[7]</a>:
</p>

<ul>
  <li><strong>Gradient-based methods</strong>: estimate importance using local gradients (e.g., Gradient × Input <a href="#gradient-input">[8]</a>).</li>
  <li><strong>Perturbation-based methods</strong>: assess how the prediction changes when individual input features are modified (e.g., SHAP <a href="#shapley-values">[9]</a>).</li>
  <li><strong>Attention-based methods</strong>: use attention weights to trace how information flows through the model (e.g., Attention Rollout <a href="#attention-rolout">[10]</a>).</li>
</ul>

<p>
While attention-based techniques may appear particularly suitable for Transformers, research has shown that attention weights are not always reliable indicators of feature importance <a href="#att1">[11]</a><a href="#att2">[12]</a>. As a result, gradient-based techniques remain among the most widely used approaches, largely due to their computational efficiency compared to perturbation-based techniques. Yet, it's worth noting that these methods were originally designed for simpler architectures, and may not be fully adequate when applied to Transformers.
</p>

<p>
This brings up a fundamental question: <strong>Are existing attribution methods truly suitable for interpreting Transformer models?</strong> 

A crucial property that any reliable attribution technique should uphold is <strong>conservation</strong> (also known as <em>completeness</em>) — the principle that the sum of all attributions should match the difference in the model’s output between the actual input and a chosen baseline (a neutral or uninformative input, used as a reference point to isolate the effect of each input feature, such as a black image in image classification tasks).
</p>


<h3 id="why-conservation">2. Why conservation is crucial to build XAI?</h3>
<p>

Without enforcing conservation, attribution-based explanations can become misleading — either by missing important input contributions or by exaggerating the relevance of unimportant ones. This issue is especially critical for complex architectures like Transformers, where components such as attention mechanisms and layer normalization are known to distort the flow of relevance through the network. Rather than proposing a new attribution score, the paper focuses on how to propagate existing attributions through the model in a way that strictly preserves conservation. In other words, the authors study how to ensure that, at each layer of the Transformer, the total relevance is neither lost and not artificially created.

The diagram below shows how different students contributed to a group blog project across three main tasks: content creation, writing, and coding. To evaluate each student's contribution to the final product, we work backward, tracing the completed blog back to the individual tasks based on their importance, and further attributing each task to the students who contributed. Doing so, we observe that the final result can mainly be attributed to Student 3, while Student 1 and 2 played smaller roles. This backward step-by-step attribution ensures that every contribution is accounted for without any distortion, preventing effort from being lost or exaggerated along the way. This is the idea behind Layer-wise Relevance Propagation (LRP), described in the following part.
</p>

<iframe src="/illustrations_XAI/uneven_blog_sankey.html" width="100%" height="500" frameborder="0"></iframe>

<h3 id="LRP">3. Layer-wise Relevance Propagation (LRP) method</h3>
<p>Layer-wise Relevance Propagation (LRP) is a method developed to explain the predictions of neural networks by attributing relevance scores to input features. It works by propagating the model’s output backward through the network, redistributing the prediction layer by layer until the input is reached. One of its main advantages is that it satisfies the conservation principle: the total relevance remains constant at each step of the propagation. Originally developed for standard deep neural networks, LRP must be adapted to handle the specific challenges posed by Transformers, such as attention mechanisms. Before addressing these adaptations, let’s first review how the basic version of LRP works.
 </p>

<h4 id="conservation-breaks">3.1 Understanding how relevance is propagated and where conservation fails</h4>
<p>
<p>
To understand whether a model satisfies conservation, we must analyze how relevance flows through each layer. The process begins by assigning all the output relevance to the predicted class only. Formally, we define the relevance vector at the final layer $L$, denoted $r_i^{(L)}$, such that:
</p>

$$
r_i^{(L)} = 
\begin{cases}
F_i(x) & \text{if } i \text{ is the predicted class}, \\\\
0 & \text{otherwise}
\end{cases}
$$

<p>
From there, the relevance is redistributed backward through the network, following specific propagation rules (e.g., LRP-γ, LRP-ε, or LRP-0). One possible rule is the <strong>Gradient × Input</strong> method, which attributes relevance based on the gradient of the output with respect to each input, scaled by the input itself:
</p>

$$
R(x_i) = x_i \frac{\partial f}{\partial x_i}, \quad R(y_j) = y_j \frac{\partial f}{\partial y_j}
$$

<p>
By applying the chain rule, this becomes:
</p>

$$
R(x_i) = \sum_{j} y_j \frac{\partial y_j}{\partial x_i} R(y_j)
$$

<p>
This formulation allows us to analyze the propagation of relevance and check whether conservation holds at each layer. Specifically, we say that a propagation rule is <strong>locally conservative</strong> if the sum of relevance scores remains constant from one layer to the next:
</p>

$$
\sum_i R(x_i) = \sum_j R(y_j)
$$

<p>
If this equality is maintained throughout the entire network — from the output all the way back to the input — then the method is said to be <strong>globally conservative</strong>. When the rule fails to preserve this equality at any layer, we say that conservation breaks, and the explanation becomes less trustworthy.
</p>


<h4 id="apply-transformers">3.2 Apply directly to transformer architectures?</h4>
<p>
When applying relevance propagation to Transformer architectures, the conservation principle is not always preserved. The paper identifies two key components where conservation systematically fails and where standard propagation rules require adaptation: Attention Heads and Layer Normalization.
</p>

<!-- Attention Heads -->
<p><strong>Propagation through Attention Heads:</strong></p>

<p>
The figure below illustrates a standard attention head, where relevance \( \mathcal{R}(y) \) is propagated backward through the attention mechanism. This includes a bilinear transformation followed by a softmax over the key-query scores. The authors demonstrate that conservation typically breaks in this setting: some attention heads receive too much relevance, while others are undervalued. This leads to distorted explanations that do not faithfully reflect the model’s true internal computations.
</p>

<p align="center">
  <img src="/images/illustrations_XAI/attention_heads.png" alt="Attention head propagation">
</p>

<!-- LayerNorm -->
<p><strong>Propagation through Layer Normalization:</strong></p>

<p>
In the case of Layer Normalization, the focus is on the centering and scaling operations applied to the inputs. These include subtracting the mean and dividing by the norm of the input — operations that inherently distort the distribution of relevance. The authors show that, regardless of the propagation rule used, conservation is systematically violated when passing through this layer. In other words, relevance is either lost or created during normalization, which undermines the reliability of the explanation.
</p>

<p align="center">
  <img src="/images/illustrations_XAI/layer_norm.png" alt="LayerNorm propagation">
</p>

<p>
These findings show that classical LRP rules cannot be directly applied to Transformers without modification. Addressing these structural issues is necessary for building explanation methods that preserve conservation and provide trustworthy insights into model behavior.
</p>

<h3 id="fixing-breaks">4. Fixing conservation breaks: a simple but effective trick</h3>

<p>
To restore conservation in Transformers, the authors propose a solution: instead of redesigning a new attribution method from scratch, they adjust how existing methods are applied by introducing a <strong>locally linear approximation</strong> of the attention heads and LayerNorm layers. This trick allows them to reuse the rules from LRP while preserving theoretical soundness.
</p>

<h4 id="Attention-Heads">4.1 Locally linear expansion for Attention Heads</h4>

<p>
During explanation time, the attention mechanism
$$ y_j = \sum_i x_i \, p_{ij} $$
is approximated by treating the attention weights \( p_{ij} \) as fixed constants (they normally depend on the input). This means we "freeze" them so that no gradient is propagated through the softmax. The attention head is then seen as a simple linear layer with fixed coefficients, and the relevance can be propagated using the following LRP rule:
</p>

$$
\mathcal{R}(x_i) = \sum_j \frac{x_i \, p_{ij}}{\sum_{i'} x_{i'} \, p_{i'j}} \, \mathcal{R}(y_j) \quad \text{(AH-rule)}
$$

<p>
This linearization not only restores conservation but also simplifies the computation, as no gradients need to flow through the attention scores.
</p>

<h4 id="LayerNorm">4.2 Locally linear expansion for LayerNorm</h4>

<p>
LayerNorm applies a normalization step that shifts and scales the input:
</p>

$$
y_i = \frac{x_i - \mathbb{E}[x]}{\sqrt{\varepsilon + \mathrm{Var}[x]}}
$$

<p>
Here too, the trick is to <strong>freeze the normalization factor</strong> \( \alpha = \frac{1}{\sqrt{\varepsilon + \mathrm{Var}[x]}} \). Once this is done, the transformation becomes linear again, and can be expressed as:
</p>

$$
y = \alpha Cx, \quad \text{where} \quad C = I - \frac{1}{N} \mathbf{1}\mathbf{1}^\top
$$

<p>
The corresponding relevance rule (<strong>LN-rule</strong>) is then:
</p>

$$
\mathcal{R}(x_i) = \sum_j \frac{x_i \, C_{ij}}{\sum_{i'} x_{i'} \, C_{i'j}} \, \mathcal{R}(y_j) \quad \text{(LN-rule)}
$$

<p>
Freezing these components essentially allows the explanation to bypass their non-linearities, making the relevance propagation both tractable and faithful to the model's internal behavior.
</p>

<h4 id="implementation">4.3 Implementation made easy</h4>

<p>
The best part about this method ? This strategy is remarkably simple to implement. In practice, you don’t need to rewrite custom backward rules. All you need to do is freeze the components during the forward pass using the <code>.detach()</code> function in PyTorch. For example:
</p>

<ul>
  <li>Replace \( p_{ij} \) with <code>p_{ij}.detach()</code> inside attention layers</li>
  <li>Freeze \( \sqrt{\varepsilon + \mathrm{Var}[x]} \) in LayerNorm by detaching it</li>
</ul>

<p>
Then, you can run your usual <em>Gradient × Input</em> attribution as usual, except that now, the relevance propagation respects conservation and produces more trustworthy explanations. As a bonus, computation is faster since gradients no longer need to be computed through these detached components.
</p>

<p>
This implementation trick, though minimal, has a major impact: it transforms Gradient × Input from a noisy, non-conservative method into a principled, conservation-respecting explanation technique for Transformers.
</p>

<h3 id="experiments">5. Confirmation with Experiments</h3>

<p>
So, does this new way of propagating relevance through Transformers actually work better? The authors ran a bunch of experiments to find out — and the short answer is: <strong>yes, absolutely</strong>.
</p>

<p>
The method  (<strong>LRP (AH+LN)</strong>) was tested against several well-known explanation techniques across a variety of tasks. We're talking about:
</p>

<ul>
  <li><strong>Text classification</strong> : movie reviews (IMDB), sentiment tweets, and SST-2</li>
  <li><strong>Image recognition</strong> : handwritten digits with MNIST</li>
  <li><strong>Molecular prediction</strong> : predicting biochemical properties from molecule data (BACE)</li>
</ul>

<p>
To evaluate how good the explanations were, they looked at two key aspects:
</p>
<ul>
  <li><strong>Quantitative metrics</strong>: How well does the explanation match the model’s actual reasoning?</li>
  <li><strong>Qualitative impressions</strong>: Are the explanations clear, focused, and free from noise?</li>
</ul>

<h4 id="quantitative-results">5.1 Quantitative Results</h4>

<p>
For the quantitative evaluation, the authors use <strong>AUAC</strong> (Area Under the Activation Curve), that measures how well an explanation highlights the most relevant parts of the input, according to the model’s own internal behavior. The AUAC metric is computed from evaluation setups where only the most (or least) relevant parts of the input are kept, to test how well explanations reflect the model's decision-making. In this context, a higher AUAC indicates a more faithful and precise explanation.
</p>

<style>
  table.auac-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1em;
  }

  table.auac-table th,
  table.auac-table td {
    border: 1px solid #ddd;
    padding: 10px 16px;
    text-align: center;
  }

  table.auac-table th {
    background-color: #f2f2f2;
  }

  table.auac-table td:first-child {
    text-align: left;
  }

  table.auac-table tr:last-child {
    background-color: #fdf5e6;
    font-weight: bold;
  }
</style>

<table class="auac-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>IMDB</th>
      <th>SST-2</th>
      <th>BACE</th>
      <th>MNIST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random</td>
      <td>0.673</td>
      <td>0.664</td>
      <td>0.624</td>
      <td>0.324</td>
    </tr>
    <tr>
      <td>Attention (last)</td>
      <td>0.708</td>
      <td>0.712</td>
      <td>0.620</td>
      <td>0.862</td>
    </tr>
    <tr>
      <td>Rollout</td>
      <td>0.738</td>
      <td>0.713</td>
      <td>0.653</td>
      <td>0.358</td>
    </tr>
    <tr>
      <td>GAE</td>
      <td>0.872</td>
      <td>0.821</td>
      <td>0.675</td>
      <td>0.426</td>
    </tr>
    <tr>
      <td>GI (Gradient × Input)</td>
      <td>0.920</td>
      <td>0.847</td>
      <td>0.646</td>
      <td>0.942</td>
    </tr>
    <tr>
      <td><strong>LRP (AH+LN)</strong></td>
      <td><strong>0.939</strong></td>
      <td><strong>0.908</strong></td>
      <td><strong>0.707</strong></td>
      <td><strong>0.948</strong></td>
    </tr>
  </tbody>
</table>

<p>
These results show that <strong>LRP (AH+LN)</strong> not only preserves theoretical properties like conservation, but also translates into superior empirical performance across tasks ranging from sentiment analysis to molecular prediction.
</p>


<h4 id="qualitative-results">5.2 Qualitative Comparison</h4>

<p>
Beyond metrics like AUAC, the authors also examine how different explanation methods behave in practice — both on language (SST-2) and vision (MNIST) tasks. In particular, they visualize how each method highlights relevant input features, and compare their interpretability and focus.
</p>

<p>
On the SST-2 dataset, all methods correctly assign relevance to the words “best” and “virtues” in a positively labeled sentence. However, <strong>A-Last</strong> overly emphasizes the word “eastwood”, suggesting an undesirable bias toward named entities. In contrast, <strong>LRP (AH)</strong> and <strong>LRP (AH+LN)</strong> assign lower relevance such entity tokens and focus more on sentiment-related words — resulting in more robust and generalizable explanations.
</p>

<p>
On MNIST (Graphormer model), the same pattern holds: <strong>LRP (AH+LN)</strong> better localizes the relevance onto the digit-containing superpixels, while attention-based methods like <strong>Rollout</strong> tend to spread relevance into the background. This confirms that the proposed method yields more precise and informative visualizations.
</p>

<style>
  table.qualitative-results {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1em;
    font-size: 15px;
  }

  table.qualitative-results th, 
  table.qualitative-results td {
    border: 1px solid #ddd;
    padding: 12px 18px;
    text-align: center;
  }

  table.qualitative-results th {
    background-color: #f9f9f9;
    font-weight: bold;
  }

  table.qualitative-results td:first-child {
    text-align: left;
  }

  table.qualitative-results tr.highlight {
    background-color: #fff8dc;
    font-weight: bold;
  }
</style>

<table class="qualitative-results">
  <thead>
    <tr>
      <th>Method</th>
      <th>Interpretability</th>
      <th>Focus on Relevant Inputs</th>
      <th>Entity/Background Bias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A-Last</td>
      <td>Moderate</td>
      <td>Moderate</td>
      <td>High</td>
    </tr>
    <tr>
      <td>Rollout</td>
      <td>Moderate</td>
      <td>Low–Moderate</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <td>GI (Gradient × Input)</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>LRP (AH)</td>
      <td>High</td>
      <td>High</td>
      <td>Low</td>
    </tr>
    <tr class="highlight">
      <td><strong>LRP (AH+LN)</strong> <br><em>(proposed)</em></td>
      <td><strong>Very High</strong></td>
      <td><strong>Very High</strong></td>
      <td><strong>Very Low</strong></td>
    </tr>
  </tbody>
</table>

<p>
These qualitative results support the idea that conservation-aware relevance propagation leads to sharper, more focused explanations, and avoids biases toward irrelevant tokens or background noise.
</p>


<h3 id="conclusion">6. Key Takeaway</h3>
<p>
Interpreting Transformer models is not straightforward — standard attribution methods often fail to accurately trace relevance through components like Attention Heads and LayerNorm. In this blog post, we explored a targeted solution introduced by Ali et al., which consists in approximating these components as locally linear during explanation.
</p>

<p>
This small conceptual shift restores the conservation of relevance and significantly improves the quality of explanations. The results speak for themselves: more precise, less noisy, and more interpretable relevance maps across a wide range of tasks.
</p>


<h3 id="references">References</h3>

<p id="health1">[1] Hörst et al. (2023). CellViT: Vision Transformers for Precise Cell Segmentation and Classification. 
Available <a href="https://arxiv.org/abs/2306.15350"><strong>here</strong></a>.</p>

<p id="health2">[2] Boulanger et al. (2024). Using Structured Health Information for Controlled Generation of Clinical Cases in French. 
Available <a href="https://aclanthology.org/2024.clinicalnlp-1.14.pdf"><strong>here</strong></a>.</p>

<p id="cyber1">[3] Seneviratne et al. (2022). Self-Supervised Vision Transformers for Malware Detection. 
Available <a href="https://arxiv.org/abs/2208.07049"><strong>here</strong></a>.</p>

<p id="cyber2">[4] Omar and Shiaeles. (2024). VulDetect: A Novel Technique for Detecting Software Vulnerabilities Using Language Models. 
Available <a href="https://pure.port.ac.uk/ws/portalfiles/portal/80445773/VulDetect_A_novel_technique_for_detecting_software_vulnerabilities_using_Language_Models.pdf"><strong>here</strong></a>.</p>

<p id="recr">[5] Aleisa, Monirah Ali; Beloff, Natalia; White, Martin (2023). Implementing AIRM: A new AI recruiting model for the Saudi Arabia labour market, Journal of Innovation and Entrepreneurship. Available <a href="https://www.econstor.eu/bitstream/10419/290242/1/1884179207.pdf"><strong>here</strong></a>.</p></p>

<p id="edu">[6] Guo, K., Wang, D. (2024) To resist it or to embrace it? Examining ChatGPT’s potential to support teacher feedback in EFL writing. Available <a href="https://link.springer.com/article/10.1007/s10639-023-12146-0"><strong>here</strong></a>.</p>

<p id="integrated-gradients">[7] Mukund Sundararajan and Ankur Taly and Qiqi Yan (2017) Axiomatic Attribution for Deep Networks. Available <a href="https://arxiv.org/pdf/1703.01365"><strong>here</strong></a>.</p>

<p id="gradient-input">[8] Avanti Shrikumar and Peyton Greenside and Anna Shcherbina and Anshul Kundaje (2017) Not Just a Black Box: Learning Important Features Through Propagating Activation Differences. Available <a href="https://arxiv.org/pdf/1605.01713"><strong>here</strong></a>.</p>

<p id="shapley-values">[9] Scott Lundberg and Su-In Lee (2017) A Unified Approach to Interpreting Model Predictions. Available <a href="https://arxiv.org/pdf/1705.07874"><strong>here</strong></a>.</p>

<p id="attention-rolout">[10] Samira Abnar and Willem Zuidema (2020) Quantifying Attention Flow in Transformers. Available <a href="https://arxiv.org/pdf/2005.00928"><strong>here</strong></a>.</p>

<p id="att1">[11] Sarthak Jain and Byron C. Wallace (2019) Attention is not Explanation. Available <a href="https://aclanthology.org/N19-1357.pdf"><strong>here</strong></a>.</p>

<p id="att2">[12] Sofia Serrano and Noah A. Smith (2019) Is Attention Interpretable? Available <a href="https://arxiv.org/pdf/1906.03731"><strong>here</strong></a>.</p>

<script type="text/x-mathjax-config">
   MathJax.Hub.Config({
       tex2jax: {
           inlineMath: [['$', '$'], ['\\(', '\\)']],
           displayMath: [['$$','$$']],
           skipTags: ['script', 'noscript', 'style', 'textarea', 'pre']
       },
       "HTML-CSS": { linebreaks: { automatic: true } }
   });
</script>
<script type="text/javascript"
        async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-AMS_HTML-full">
</script>

</body>
</html>

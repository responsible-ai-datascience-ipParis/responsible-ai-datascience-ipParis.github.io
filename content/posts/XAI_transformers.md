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

While models relying on Transformers have shown impressive performance, their behavior remains hard to explain, raising questions about their use in sensitive domains like healthcare <a href="#health1">[1]</a>, <a href="#health2">[2]</a>, cybersecurity [cyber1, cyber2], recruitment [recr.] or education [edu]. Understanding their decisions therefore becomes a major challenge, to ensure that they do not discriminate on unwanted features (eg. gender, ethnicity). </p>

<p align="center">
  <img src="/images/illustrations_XAI/illustration_intro.png" alt="illustration_intro">
</p>

<!-- TABLE OF CONTENTS -->
<div class="toc">
   <strong>Table of Contents</strong>
   <ul>
      <li><a href="#attribution-methods">1. Attribution methods: how to identify important features?</a></li>
      <li><a href="#why-conservation">2. Why conservation is crucial to build XAI?</a></li>
      <li><a href="#LRP">3. Layer-wise Relevance Propagation (LRP) method</a>
         <ul>
            <li><a href="#conservation-breaks">3.1 How can we detect where the conservation breaks?</a></li>
            <li><a href="#apply-transformers">3.2 Apply directly to transformer architectures?</a></li>
         </ul>
      </li>
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
Numerous attribution techniques have been proposed, each relying on different strategies to assess feature importance. As defined in <a href="#integrated-gradients">Integrated Gradients</a>, these methods generally fall into three categories:
</p>

<ul>
  <li><strong>Gradient-based methods</strong>: estimate importance using local gradients (e.g., <a href="#gradient-input">Gradient × Input</a>).</li>
  <li><strong>Perturbation-based methods</strong>: assess how the prediction changes when individual input features are modified (e.g., <a href="#shapley-values">SHAP</a>).</li>
  <li><strong>Attention-based methods</strong>: use attention weights to trace how information flows through the model (e.g., <em>Attention Rollout</em>).</li>
</ul>

<p>
While attention-based techniques may appear particularly suitable for Transformers, research has shown that attention weights are not always reliable indicators of feature importance <a href="#att1">[att1]</a>, <a href="#att2">[att2]</a>. As a result, gradient-based techniques remain among the most widely used approaches, largely due to their computational efficiency compared to perturbation-based techniques. Yet, it's worth noting that these methods were originally designed for simpler architectures, and may not be fully adequate when applied to Transformers.
</p>

<p>
This brings up a fundamental question: <strong>Are existing attribution methods truly suitable for interpreting Transformer models?</strong> 

A crucial property that any reliable attribution technique should uphold is <strong>conservation</strong> (also known as <em>completeness</em>) — the principle that the sum of all attributions should match the difference in the model’s output between the actual input and a chosen baseline (a neutral or uninformative input, used as a reference point to isolate the effect of each input feature, such as a black image in image classification tasks).
</p>


<h3 id="why-conservation">2. Why conservation is crucial to build XAI?</h3>
<p>

Without enforcing conservation, attribution-based explanations can become misleading — either by missing important input contributions or by exaggerating the relevance of unimportant ones. This issue is especially critical for complex architectures like Transformers, where components such as attention mechanisms and layer normalization are known to distort the flow of relevance through the network. Rather than proposing a new attribution score, the paper focuses on how to propagate existing attributions through the model in a way that strictly preserves conservation. In other words, the authors study how to ensure that, at each layer of the Transformer, the total relevance is neither lost and not artificially created.

The diagram below shows how different students contributed to a group blog project across three main tasks: content creation, writing, and coding. We can see that Student 3 provided the vast majority of the work in all stages, while Student 1 and Student 2 made only minor contributions. Each task — whether it be content, writing, or code — is then aggregated into the final blog. This flow highlights the importance of tracking and preserving contributions throughout the process: if some efforts were lost or inflated along the way, the final picture would not reflect reality. In the same way, when explaining model predictions, preserving the total “relevance” as it moves through each layer ensures that the explanation remains faithful to the model’s actual decision process.
</p>

<iframe src="/illustrations_XAI/uneven_blog_sankey.html" width="100%" height="500" frameborder="0"></iframe>

<h3 id="LRP">3. Layer-wise Relevance Propagation (LRP) method</h3>
<p>Layer-wise Relevance Propagation (LRP) is a method developed to explain the predictions of neural networks by attributing relevance scores to input features. It works by propagating the model’s output backward through the network, redistributing the prediction layer by layer until the input is reached. One of its main advantages is that it satisfies the conservation principle: the total relevance remains constant at each step of the propagation. Originally developed for standard deep neural networks, LRP must be adapted to handle the specific challenges posed by Transformers, such as attention mechanisms. Before addressing these adaptations, let’s first review how the basic version of LRP works.
 </p>

<h4 id="conservation-breaks">3.1 Understanding how relevance is propagated and where conservation fails</h4>
<p>
<p>
To understand whether a model satisfies conservation, we must analyze how relevance flows through each layer. The process begins by assigning all the output relevance to the predicted class only. Formally, we define the relevance vector at the final layer $L$, denoted $r^{(L)}$, such that:
</p>

$$
r_i^{(L)} = 
\begin{cases}
F_i(x) & \text{if } i \text{ is the predicted class}, \\\\
0 & \text{otherwise}
\end{cases}
$$

<p>
From there, the relevance is redistributed backward through the network, following specific propagation rules (e.g., LRP-γ, LRP-ε, or LRP-0). One common example is the <strong>Gradient × Input</strong> method, which attributes relevance based on the gradient of the output with respect to each input, scaled by the input itself:
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


<h3 id="designed-solutions">4. Designed solutions</h3>
<p>Authors then proposed propagation rules that are conservative by design, taking as a starting point the [formula](chain-rule).</p>

<h3 id="references">References</h3>

<p id="health1">[1] Hörst et al. (2023). CellViT: Vision Transformers for Precise Cell Segmentation and Classification. 
Available <a href="https://arxiv.org/abs/2306.15350"><strong>here</strong></a>.</p>

<p id="health2">[2] Boulanger et al. (2024). Using Structured Health Information for Controlled Generation of Clinical Cases in French. 
Available <a href="https://aclanthology.org/2024.clinicalnlp-1.14.pdf"><strong>here</strong></a>.</p>

<p id="cyber1">[cyber1] Seneviratne et al. (2022). Self-Supervised Vision Transformers for Malware Detection. 
Available <a href="https://arxiv.org/abs/2208.07049"><strong>here</strong></a>.</p>

<p id="cyber2">[cyber2] Omar and Shiaeles. (2024). VulDetect: A Novel Technique for Detecting Software Vulnerabilities Using Language Models. 
Available <a href="https://pure.port.ac.uk/ws/portalfiles/portal/80445773/VulDetect_A_novel_technique_for_detecting_software_vulnerabilities_using_Language_Models.pdf"><strong>here</strong></a>.</p>

<p id="recr">[recr.] Aleisa, Monirah Ali; Beloff, Natalia; White, Martin (2023). Implementing AIRM: A new AI recruiting model for the Saudi Arabia labour market, Journal of Innovation and Entrepreneurship.</p>



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

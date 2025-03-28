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
      <img src="/images/ip-logo.png" alt="Foucauld Estignard">
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

A MODIF QUAND J'AI LA CONFIRMATION For example, if a neural network classifies an image as a cat with a confidence score of 0.9, a well-designed attribution method should distribute this score among relevant pixels (e.g., 0.6 for the cat’s face and 0.3 for its body). If the attributions sum to a different value (e.g., 0.7 or 1.1), it means some contributions were lost or artificially added, making the explanation unreliable. Conservation ensures that every contributing input is accounted for properly, leading to meaningful and trustworthy explanations. 
</p>

<iframe src="/illustrations_XAI/blog_sankey.html" width="100%" height="500" frameborder="0"></iframe>

<h3 id="LRP">3. Layer-wise Relevance Propagation (LRP) method</h3>
<p>The Layer-wise Relevance Propagation (LRP) method is designed to ensure proper conservation not only at the global level but also between each pair of layers, by redistributing the output of the model layer after layer until reaching the input layer. This method was previously designed for deep neural networks. The challenge is to extend these techniques to Transformers, whose complexity (e.g. due to attention mechanisms) requires adjustments to ensure reliable explanations while respecting this conservation axiom. But first, let’s have a look at how the default LRP method works. </p>

<h4 id="conservation-breaks">3.1 How can we detect where the conservation breaks?</h4>
<p>

To ensure proper conservation, we start by setting to 0 all the non-predicted scores, defining $ r^{(L)} $, the vector of relevance scores of the deepest layer $ L $, as a vector with a single non-zero component:

$$
r_i^{(L)} = 
\begin{cases}
F_i(x) & \text{if } i \text{ is the target class}, \\
0 & \text{otherwise}
\end{cases}
$$

Then, the redistribution is done by following a predefined propagation rule (e.g., LRP-γ, LRP-ε, LRP-0). One possible rule is to use **Gradient × Input (GI)**, which defines attributions as:

$$
R(x_i) = x_i \frac{\partial f}{\partial x_i}, \quad R(y_j) = y_j \frac{\partial f}{\partial y_j}
$$

Using the [chain rule](#chain-rule), we have:

$$
R(x_i) = \sum_{j} y_j \frac{\partial y_j}{\partial x_i} R(y_j)
$$

By reformulating GI (Gradient*Input) as a relevance propagation method, we can identify where conservation breaks down in the network. Once these weaknesses have been identified, it is possible to develop better rules to ensure a more accurate explanation. The conservation axiom is respected if the sum of the relevance attributed to the inputs is equal to the sum of the relevance attributed to the outputs: 

$$
\sum_i R(x_i) = \sum_j R(y_j)
$$

If this equality is respected at each layer/component, then GI is **locally conservative**, if the equality is respected at all layers, then GI is **globally conservative**.</p>

<h4 id="apply-transformers">3.2 Apply directly to transformer architectures?</h4>
<p>For transformers architecture, it appears that two components break the conservation rule and require an improvement in the propagation rule.

- Propagation in Attention Heads: 
<p align="center">
  <img src="/images/ip-logo.png" alt="ip paris logo">
</p>
It has been shown mathematically in the paper that conservation rule breaks in most cases. This means that some attention heads can be over or under represented in the explanation, which is regrettable. 

- Propagation in LayerNorm:
<p align="center">
  <img src="/images/ip-logo.png" alt="ip paris logo">
</p>

Here authors only focused on the centering and standardization parts. They showed that for this component, conservation is never satisfied. 
</p>

<img src="/images/ip-logo.png" alt="Attention Head diagram" class="centered">
<img src="/images/ip-logo.png" alt="LayerNorm diagram" class="centered">

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

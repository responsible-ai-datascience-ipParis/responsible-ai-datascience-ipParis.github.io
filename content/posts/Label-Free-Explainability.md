+++
title = 'Label-Free Explainability'
date = 2024-03-17T15:31:34+01:00
draft = false
+++



<h1 style="font-size: 36px;">Label-Free Explainability for Unsupervised Models</h1>

<h1 style="font-size: 18px;">Authors: <a href="https://github.com/Valentinahxu">Valentina Hu </a> and  <a href="https://github.com/selmazrg"> Selma Zarga</a></h1>

# Table of Contents

- [Introduction](#section-0)
- [Experiment](#section-5)


## Why do we need explainability ? {#section-0}


Machine learning models are becoming increasingly capable of making advanced predictions. While models like linear regression are relatively easy to understand and explain, more complex models, often called **"black boxes"** due to their complexity, present challenges in explaining how they make predictions. These models can be problematic in highstakes applications such as healthcare, finance, and justice, where it's crucial to justify decision-making. Additionally, in case of errors, it's important to understand the origin in order to address and correct them.


<center>

"**Explainability is the cornerstone of trust in black box models; without it, they remain inscrutable and unreliable.**" - *Yoshua Bengio*

</center>


To tackle this challenge, the field of Explainable Artificial Intelligence (XAI) has emerged, offering various methods to enhance **model transparency**. **Post-Hoc explainability** methods exist, which intervene after the model has generated its results, enabling users to comprehend the reasoning behind specific decisions or predictions. These methods supplement the predictions of black box models with diverse explanations of how they arrive at their predictions.

![XAI explainability](/images/explainability/blackboxpng.webp)

While much of the work on explainability focuses on supervised models, where labels are available to interpret predictions, unsupervised learning models are trained without labels. This post will focus on these unsupervised, label-free cases. It is based on recent research conducted by Crabbé and van der Schaar in 2022, which explores the explainability of unsupervised models. They have developed two new methods to explain these complex models without labels. The first method highlights important features in the data, while the second identifies training examples that have the greatest impact on the model's construction of representations.



## Experiments: Evaluation and Results {#section-5}

### Overview {#section-111}

In this section, we provide an overview of the experiment conducted to evaluate the label-free extensions of various explanation methods for unsupervised models. The experiment is divided into two main parts: consistency checks and comparisons of representations learned from different pretext tasks.






### References

1. Crabbé, J. &amp; van der Schaar, M.. (2022). Label-Free Explainability for Unsupervised Models. <i>Proceedings of the 39th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 162:4391-4420 Available from https://proceedings.mlr.press/v162/crabbe22a.html.


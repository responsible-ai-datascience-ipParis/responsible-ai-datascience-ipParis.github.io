+++
title = "MaNo: A Smarter Way to Estimate Model Accuracy Without Labels"
date = 2025-02-10T18:25:03+01:00
draft = false
authors = ["Alice Devilder", "Sibylle Degos"]
affiliations = ["IP Paris, Responsible AI"]
toc = true
+++

<!-- Custom CSS for MathJax and Tables -->
<style type="text/css">
code.has-jax { 
    font: inherit;
    font-size: 100%; 
    background: inherit; 
    border: inherit;
}

table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    padding: 8px;
    text-align: center;
    border-bottom: 1px solid #ddd;
}
th {
    background-color: #f2f2f2;
}
tr:hover {
    background-color: #f5f5f5;
}
</style>

<!-- MathJax Configuration -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // Removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>

<!-- Load MathJax -->
<script type="text/javascript" 
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full">
</script>

<h1 style="font-size: 28px; text-align: center;"> MaNo: A Smarter Way to Estimate Model Accuracy Without Labels </h1>

<style>
.hr-line {
    border: none;
    height: 2px;
    background-color: black;
    margin: 10px 0;
}
</style>

<hr class="hr-line">

<strong>Authors:</strong> Alice Devilder, Sibylle Degos | <strong>Affiliations:</strong> IP Paris, Responsible AI | <strong>Published:</strong> 2025-02-10

<hr class="hr-line">

## **Table of Contents**
- [Introduction](#section-0.0)
    - [Why Do Logits Matter For Generalization Performance?](#section-0.1)
    - [Why Does softmax normalisation fail to alleviate the overconfidence issues of logits-based methods?](#section-0.2)
- [Introducing MANO: A Two-Step Approach](#section-1)
    - [Normalization with Softrun](#section-1.1)
    - [Aggregation Using Matrix Norms](#section-1.2)
- [Empirical Success: MANO vs. Baselines](#section-2)
- [Applications and Future Directions](#section-3)
- [Conclusion](#section-4)

This is a blog post about the paper ***MaNo: Exploiting Matrix Norm for Unsupervised Accuracy Estimation Under Distribution Shifts***, published by *Renchunzi Xie*, *Ambroise Odonnat*, *Vasilii Feofanov*, *Weijian Deng*, *Jianfeng Zhang* and *Bo An* in November 2024 and avalaible on [arXiv](https://arxiv.org/abs/2405.18979).

---

Imagine deploying a machine learning model into the real world, only to watch its performance crumble under unpredictable data shifts. The ability to estimate model accuracy **without labeled test data** is crucial, yet remains a challenge. **MANO (Matrix Norm-based Accuracy Estimation)** presents a novel solution, leveraging **logits**—the raw outputs of a model—to infer confidence and predict accuracy in an **unsupervised manner**.

Traditional approaches rely on costly, labor-intensive ground-truth labels, making real-time evaluation impractical. MANO sidesteps this limitation through an elegant two-step process: **Softrun normalization** to calibrate logits and **matrix norm aggregation** to quantify decision boundary distances. The result? A robust, label-free accuracy estimator that outperforms existing methods across a range of architectures and distribution shifts.

This blog dives deep into the inner workings of MANO, uncovering the theoretical foundations and empirical success that make it a game-changer in **unsupervised accuracy estimation**.

---

## **Introduction** {#section-0.0}
A common method for estimating accuracy without labels is analyzing a model’s **logits**—the raw outputs before softmax. However, existing methods suffer from **overconfidence issues** and **biased predictions** under distribution shifts.

### **Why Do Logits Matter?** {#section-0.1}
Logits, the raw model outputs before softmax transformation, encode essential information about a model’s confidence. Under the **Low-Density Separation (LDS) Assumption**, decision boundaries should lie in low-density regions, meaning that logits inherently reflect a model’s **generalization performance**.

Mathematically, for a given input $x$, the model computes logits as:

$$ q = f(x) = (\omega_k^T z)_k \in \mathbb{R}^K $$

where $z$ is the learned feature representation, $\omega_k$ is the classifier’s weight vector, and $K$ is the number of classes. The magnitude of logits correlates with the **distance to decision boundaries**, making them valuable for accuracy estimation.

### **Why Does softmax normalisation fail to alleviate the overconfidence issues of logits-based methods?** {#section-0.2}
Most traditional accuracy estimators use **softmax normalization** to transform logits into probabilities. However, softmax is **highly sensitive to outliers** and can amplify **overconfidence issues**, leading to **biased accuracy predictions**. This happens because the softmax function applies an exponential transformation, making minor logit differences appear much more significant:

$$ \text{softmax}(q_k) = \frac{e^{q_k}}{\sum_{j=1}^{K} e^{q_j}} $$

This normalization can lead to overconfident predictions, especially when the model is miscalibrated under distribution shifts.

---

## **Introducing MANO: A Two-Step Approach** {#section-1}
MANO addresses these challenges through a **two-step process**: **Softrun normalization** and **aggregation using matrix norms**.

### **1. Normalization with Softrun** {#section-1.1}
Softmax has long been the standard for transforming logits into probabilities, but its fundamental flaw lies in its **sensitivity to large logits**, which causes models to be overconfident in their predictions. This issue is particularly detrimental under distribution shifts, where incorrect predictions may be assigned excessively high probabilities, leading to biased accuracy estimates. The exponential nature of softmax exaggerates differences between logits, making the model appear more confident than it actually is.

Recognizing these shortcomings, the creators of MANO devised **Softrun**, a normalization method that mitigates overconfidence while preserving useful information. Instead of applying an unregulated exponential function, Softrun **dynamically adjusts its transformation based on dataset-wide confidence criteria**. It does so using a two-case function:

$$ \sigma(q) = \frac{v(q)}{\sum_{k=1}^{K} v(q_k)} $$

where:

$$v(q) =
\begin{cases} 
  \begin{aligned}
    1 + q + \frac{q^2}{2}, & \quad \text{if } \Phi(D_{test}) \leq \eta \quad \text{(Taylor approx.)}
  \end{aligned} \\
  \begin{aligned}
    e^q, & \quad \text{otherwise (softmax)}
  \end{aligned}
\end{cases}
$$

This formulation ensures that when the dataset is **poorly calibrated**, i.e., when the model's predictions are unreliable, Softrun applies a **Taylor approximation** rather than an exponential function. The Taylor approximation smooths out the effect of large logits, preventing the model from being overly confident in any particular prediction. By contrast, when the dataset is **well-calibrated**, the function behaves like softmax, preserving probability distributions where confidence is warranted.

The key advantage of Softrun is that it prevents **overconfidence accumulation**, a problem that softmax inherently suffers from. By adapting to the dataset's calibration quality, Softrun provides **more stable and realistic probability distributions**, leading to **improved accuracy estimation** even in challenging scenarios. 

### **2. Aggregation Using Matrix Norms** {#section-1.2}
After normalization, MANO **aggregates** the logits using the **Lp norm** of the matrix $Q$, defined as:

$$ S(f,D_{test}) = \frac{1}{p\sqrt{NK}} \|Q\|_p = \left( \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} |\sigma(q_i)_k|^p \right)^{\frac{1}{p}} $$

where **p controls sensitivity to high-confidence predictions**. This metric effectively captures the overall **feature-to-boundary distance**, making it a reliable estimator of model accuracy.

---

## **Empirical Success: MANO vs. Baselines** {#section-2}
MANO has been extensively evaluated against **11 baseline methods** across a diverse set of neural network architectures and distribution shifts. The experiments were conducted on a range of classification tasks, including image recognition benchmarks such as CIFAR-10, CIFAR-100, TinyImageNet, and ImageNet, as well as domain adaptation datasets like PACS and Office-Home. 

The evaluation setup covered **three major types of distribution shifts**: synthetic shifts, where models were tested against artificially corrupted images; natural shifts, which involved datasets collected from different distributions than the training data; and subpopulation shifts, where certain classes or groups were underrepresented in the training data. MANO consistently outperformed existing methods in all three scenarios, demonstrating its robustness to varying degrees of domain shifts.

Unlike traditional approaches that either rely on softmax probabilities or require retraining on new distributions, MANO provides a label-free and computation-efficient accuracy estimation method that scales well across different domains. By using **Softrun normalization and matrix norm aggregation**, MANO achieves a **stronger correlation with actual accuracy**, ensuring that model performance estimates remain reliable even when faced with extreme distribution shifts.

---

## **Applications and Future Directions** {#section-3}
The ability to estimate model accuracy without labels has broad implications in AI. One crucial application is **deployment risk estimation**, where real-time insights into model reliability can be obtained without costly manual labeling. This is particularly useful for models deployed in dynamic environments, such as healthcare and autonomous systems, where distribution shifts are frequent and unpredictable.

Another important application is **dataset selection**. MANO enables the identification of challenging test sets that require further improvements, allowing researchers to focus on areas where models struggle the most. By pinpointing performance gaps, AI practitioners can fine-tune models more effectively and improve generalization.

Furthermore, MANO can assist in **calibration diagnostics**, detecting when a model is overconfident and underperforming in specific regions of the data space. By highlighting miscalibrated predictions, MANO supports better model adjustments, ensuring that confidence levels align more closely with actual accuracy.

## **Conclusion** {#section-4}
MANO represents a **significant breakthrough** in unsupervised accuracy estimation. By addressing **logit overconfidence** and introducing **Softrun normalization**, MANO provides a **scalable, robust, and theoretically grounded** approach for evaluating model accuracy under distribution shifts.

🔗 **Code available at:** [MANO GitHub Repository](https://github.com/Renchunzi-Xie/MaNo)

MANO isn’t just a step forward—it’s a leap toward **trustworthy AI deployment in the wild**!

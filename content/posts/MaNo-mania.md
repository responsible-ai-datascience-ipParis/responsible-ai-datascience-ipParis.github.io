+++
title = "MaNo: Exploiting Matrix Norm for Unsupervised Accuracy Estimation Under Distribution Shifts"
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

A paper from **Renchunzi Xie**${ }^{1}$, **Ambroise Odonnat**${ }^{2}$, **Vasilii Feofanov**${ }^{2}$, **Weijian Deng**${ }^{3}$, **Jianfeng Zhang**${ }^{2}$, **Bo An**${ }^{1}$ | **Affiliations:** ${ }^{1}$Nanyang Technological University, ${ }^{2}$Huawei Noah’s Ark Lab, ${ }^{3}$Australian National University | **Published:** 2024-11-25

---

<h1 style="font-size: 28px; text-align: center;"> MaNo-mania: Cracking the Code of Unsupervised Accuracy Estimation! </h1>

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

Machine learning models deployed in real-world scenarios often face **distribution shifts**, leading to deteriorated performance. Traditional methods for monitoring model accuracy rely on access to **ground-truth labels**, which is **resource-intensive** and **impractical**. **MANO (Matrix Norm-based Accuracy Estimation)** introduces a novel unsupervised accuracy estimation approach. Let's explore how MANO works and why it matters.



## **Table of Contents**
- [Introduction](#section-0.0)
    - [Why Do Logits Matter For Generalization Performance?](#section-0.1)
    - [Why Does softmax normalisation fail to alleviate the overconfidence issues of logits-based methods?](#section-0.2)
- [Introducing MANO: A Two-Step Approach](#section-1)
    - [Normalization with Softrun](#section-1.1)
    - [Aggregation Using Matrix Norms](#section-1.2)
- [Empirical Success: MANO vs. Baselines](#section-2)
- [Mathematical Formulation of MANO](#section-3)
- [Applications and Future Directions](#section-4)
- [Conclusion](#section-5)

---

## **Introduction** {#section-0.0}
A common method for estimating accuracy without labels is analyzing a model’s **logits**—the raw outputs before softmax. However, existing methods suffer from **overconfidence issues** and **biased predictions** under distribution shifts.

### **Why Do Logits Matter?** {#section-0.1}
Logits represent distances between learned features and the **decision boundary**. Based on the **Low-Density Separation (LDS) Assumption**, well-trained models position decision boundaries in low-density regions, ensuring that logits indicate confidence in classification.

### **Why Does softmax normalisation fail to alleviate the overconfidence issues of logits-based methods?** {#section-0.2}
Traditional methods rely on **softmax normalization** to convert logits into probabilities. However, softmax is **sensitive to outliers** and **overconfident predictions**, leading to **biased accuracy estimation**. This issue is exacerbated under **distribution shifts**.

## **Introducing MANO: A Two-Step Approach** {#section-1}
MANO employs a **two-step process**: Normalization with Softrun and Aggregation using Matrix Norms.

### **1. Normalization with Softrun** {#section-1.1}
Instead of relying on **softmax**, MANO introduces a **novel normalization function called Softrun**. When the model is well-calibrated, **Softrun behaves like softmax**. However, when confidence is unreliable, **Softrun adjusts logits to reduce overconfidence and bias**.

### **2. Aggregation Using Matrix Norms** {#section-1.2}
After normalization, MANO aggregates logits information using the **Lp norm of the logits matrix**. This metric reflects the overall distance of features to decision boundaries, **providing a standardized and robust accuracy estimation score**.

## **Empirical Success: MANO vs. Baselines** {#section-2}
Experiments demonstrate that MANO **outperforms 11 baseline methods** across various architectures and distribution shifts:

✅ **Higher correlation** with actual accuracy  
✅ **More robust** across distribution shifts  
✅ **Better calibration**, reducing overconfidence issues  

## **Mathematical Formulation of MANO** {#section-3}
We define the normalized logits matrix $X$ as follows:

$$
X = \frac{Z}{\|Z\|_p}
$$

where $Z$ represents the original logits, and $\|Z\|_p$ denotes the matrix $L_p$-norm. The final accuracy estimation score is computed as:

$$
S = \sum_{i=1}^{n} \left| \frac{X_i}{X_{max}} \right|
$$

where $X_{max}$ represents the largest logit magnitude.

## **Applications and Future Directions** {#section-4}
MANO can be applied in multiple scenarios:

- **Deployment Risk Estimation**: Assessing model reliability without labeled test data.
- **Dataset Selection**: Identifying challenging test cases.
- **Calibration Diagnostics**: Detecting overconfident predictions.

## **Conclusion** {#section-5}
MANO sets a new benchmark in **unsupervised accuracy estimation**, addressing **logit overconfidence** with a **novel normalization technique**.

🔗 **Code available at:** [GitHub Repository](https://github.com/Renchunzi-Xie/MaNo)

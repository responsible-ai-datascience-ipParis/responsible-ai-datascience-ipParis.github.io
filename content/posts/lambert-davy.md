+++
title = 'Learning Fair Scoring Functions: Bipartite Ranking under ROC-based Fairness Constraints'
date = 2024-03-23T19:39:13+01:00
draft = false
+++

<h1 style="font-size: 24px;">Learning Fair Scoring Functions: Bipartite Ranking under ROC-based Fairness Constraints</h1>

<h1 style="font-size: 18px;">Authors: Godefroy LAMBERT and Louise DAVY</h1>

# Table of Contents

- [Introduction](#section-1)
- [Basic definitions](#section-2)
- [AUC-based fairness constraints](#section-3)
- [ROC-based fairness constraints](#section-4)
- [Experiments](#section-5)
- [Conclusion](#section-6)


# Introduction {#section-1}

With recent advances in machine learning, applications are becoming increasingly numerous and the expectations are high. Those applications will only be able to be deployed if some important issues are addressed such as bias. There are famous datasets known for containing variables that induce a lot of bias such as Compas with racial bias and gender bias in the Adult dataset. To avoid those biases, new algorithms were created to provide more fairness in the prediction by using diverse methods. Today, we will be reviewing the methods presented in “Learning Fair Scoring Functions: Bipartite Ranking under ROC-based Fairness Constraints”. This paper uses basic metrics such as AUC constraint and ROC constraint and shows some limitations. Since this is bipartite ranking, we will only focus on binary prediction, such as will this person recid for the COMPAS dataset or will this person get his loan for the Adult dataset.

# Basic definitions {#section-2}

The goal of bipartite ranking is to acquire an ordering of X where positive instances are consistently ranked above negative ones with a high probability. This is done by learning an appropriate scoring function s. Such scoring functions are widely used in many critical domains such as loan granting, anomaly detection, or even in court decisions. A nice way to assess their performance is through the analysis of the Receiver Operating Characteristic (ROC) curve. It does so by plotting the true positive rate against the false positive rate at various thresholds. A curve closer to the upper-left corner represents a better-performing scoring function. To identify how close a curve is close to the uppe-left corner, we compute the Area Under the ROC Curve (AUC). 

While fairness seems like a desirable goal for any ranking function, there are many different definitions of what fairness really is and thus, many different metrics to assess the fairness of an algorithm. In the case of loan grants for example, one could consider that fairness is achieved between men and women if we granted the same percentage of loans for both groups. Statistical parity, which  compares the proportion of positive outcomes between different demographic groups, is a good metric in this case.  However, this approach might overlook underlying disparities in socioeconomic status that affect loan approval rates. Another vision of fairness might ensure that individuals are all as likely to get a wrong decision, regardless of demographic factors such as gender or ethnicity. In this case, parity of mistreatment would be a good metric, as it ensures that the proportion of errors is the same for all demographic groups. However, this considers that all errors are the same, which means that one group could have a high false positive rate and another a high false negative rate. The authors thus decided to choose parity in false positive rates and/or parity in false negative rates.

# AUC-based fairness constraints {#section-3}

Blabla 

# ROC-based fairness constraints {#section-4}

A richer approach is then to use pointwised ROC-based fairness constraints. Ideally, we would want to enforce the equality of all score distributions between both groups (i.e., identical ROC curves). This would satisfy all AUC-based fairness constraints previously mentioned. However, this condition is so restrictive that it will most likely lead to a significant drop in performances. As a result, the authors propose to satisfy this constraint on only a finite number of points. They were indeed able to prove that this was sufficient to ensure fair classifiers.

We can introduce the learning objective $L_\Lambda(s)$ defined as:
\begin{align*}
    % L_\Lambda(s) = 
    AUC_{H_s,G_s} &- 
    \sum_{k=1}^{m_H} \lambda_H^{(k)}  \big| \Delta_{H,\alpha_H^{
    (k)}}(s) \big| 
    - \sum_{k=1}^{m_G} \lambda_G^{(k)} \big| \Delta_{G,\alpha_G^{(k)}}(s) \big|,
\end{align*}


# Experiments {#section-5}

Blabla

# Conclusion {#section-6}

Blabla


# References

1. Blabla



<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>
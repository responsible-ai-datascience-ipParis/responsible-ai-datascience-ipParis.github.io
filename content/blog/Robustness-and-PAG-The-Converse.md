
+++
title = 'Robustness and Perceptually Aligned Gradients : does the converse stand ?'
date = 2024-02-07T16:06:43+01:00
draft = false

+++

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

In the context of image recognition in Machine Learning, one could quickly realize that building *robust* models is crucial. Having failures could potentially lead to worrying outcomes and it is part of the design to aim to implement models that would be prevented against ***adversarials attacks***, that will be explained. At some point, when reaching models that are robust, it somehow occurs that small variations made are easily interpretable by humans, something which is not common in current ML models such as this one. Having noticed this phenomenon, the authors of the paper would try to verify the opposite assumption. By building models that verify this idea of alignment with human perception, do we create robust models ? 

## Adversarial attacks
But before explaining the article, it could be relevant to explain briefly what are adversarial attacks and how it led to the design of robustness. 


Adversarial attacks refer to a class of techniques in machine learning where intentionally crafted input data is used to deceive or mislead a model, leading it to make incorrect predictions or classifications. These attacks exploit vulnerabilities in the model's decision-making process, taking advantage of the model's sensitivity to small changes in input data that might be imperceptible to humans.
They are most prominently associated with deep learning models, particularly neural networks, due to their high capacity and ability to learn complex patterns.

Concretly, in a theoretical framework, the usual example is to make a model classify an image of a cat as a dog or another animal, without any way for the human to notice it. However, consequences can be more dreadful in real life as one could consider what would happen if an autonomous vehicles missclassified a stop sign as speed limit sign. INSERER IMAGES + REFERENCES

Now, let's dive a bit deeper to understand how these errors happen.
Several points can be highlighted, such as the level of linearity of Neural Networks, but one acknowledged moot point dwells on the use of Loss function in Deep Learning methods. Indeed, especially when considering datasets of pictures, there are many directions where the loss is steep. It would mean that it can be highly delicate to propose a good minimization of the loss. Moreover, the main idea for our problem is that a small change of the input can cause abrupt shifts in the decision process of our model. This effect increases with the dimensionnality (quality of pictures...) and therefore will still be relevant with time.  

The basic modelisation of an attack would be the following. Let's consider :
- a model $f\ :\ \mathcal{X} \ \rightarrow \ \mathcal{Y}$
- the input to pertub : $x \in \mathcal{X}$
- a potential target label : $t \in  \mathcal{Y}$
- a small perturbation : $\eta$

Then, mathematically, the attacker would try to have something that verifies $f(x + \eta) = t$ (or any other label than $f(x)$ for an untargeted attack). 

Now, as one can imagine, it is possible to compute attacking models related to this framework. Let's understand two well-knowns algorithms that follow this goal.






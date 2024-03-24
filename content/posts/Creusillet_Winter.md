+++
title = 'Creusillet_Winter'
date = 2024-02-07T15:55:14+01:00
draft = false
+++



<h1 style="font-size: 36px;">To update or not to update? Neurons at equilibrium in deep models
</h1>

<h1 style="font-size: 24px;">Author: Alexis WINTER Augustin CREUSILLET</h1>


## Introduction

### Background

Recent advances in deep learning have undeniably propelled the field to unprecedented heights, revolutionizing various domains from computer vision to natural language processing. However, these strides forward have not come without a significant toll on computational resources. As models grow increasingly complex , the demand for computational power has surged exponentially. One of the most expensive tasks within the realm of deep learning is undoubtedly the training of models. This process entails iteratively adjusting millions or even billions of parameters to minimize a predefined loss function, requiring extensive computational power and time-intensive operations. This relentless hunger for resources poses challenges in terms of both affordability and environmental sustainability, highlighting the need for innovative solutions to make deep learning more efficient and accessible in the face of escalating computational demands.


This paper try to focus on the overall behavior of neurons, leveraging the notion of neuronal equilibrium (NEq). When a neuron reaches a state of equilibrium, indicating that it has learned a particular input-output mapping, it cease its updates. The result is that we can reduce the number of operation needed for the computation of the backpropagation and optimizer and thus reduce the amount of ressources necessary for the model.


### Related works

#### Pruning strategies

Pruning strategies consist in the systematic removal of redundant or less important parameters, connections, or units within a model to improve efficiency and reduce computational complexity. These strategies are inspired by the biological concept of pruning, where unnecessary connections in neural networks are eliminated to enhance neural efficiency. Pruning can take various forms, including magnitude-based pruning, where parameters with small weights are pruned, or structured pruning, which removes entire neurons, channels, or layers based on specific criteria. Pruning strategies effectively reduce the model size leading to a more frugal and compact model.

Despite its effectiveness in reducing model size and improving inference efficiency, pruning strategies typically do not alleviate the computational complexity associated with training neural networks. While pruning removes parameters or connections during the inference phase, the training process still requires the full model to be trained initially, often resulting in high computational demands. In fact, pruning can even increase training complexity due to the need for additional iterations to fine-tune the remaining parameters and adapt the model to compensate for the pruned components. Consequently, while pruning offers significant benefits in terms of model deployment and inference efficiency, it does not directly address the computational burden of training models.


#### Lottery ticket hypothesis

The lottery ticket hypothesis is a concept in deep learning that suggests that within a dense neural network, there exist sparse subnetworks, or "winning tickets," that are capable of achieving high accuracy when trained in isolation. These winning tickets are characterized by having a small subset of well-initialized weights, which when pruned to remove the remaining connections, can maintain or even surpass the performance of the original dense network.

The hypothesis was introduced by Jonathan Frankle and Michael Carbin in 2018. They conducted experiments demonstrating that randomly-initialized, dense neural networks contain subnetworks that can achieve high performance when trained properly. These subnetworks, or winning tickets, tend to emerge during the training process and possess a specific initialization that allows them to be effectively trained within the broader network.

The significance of the lottery ticket hypothesis lies in its potential to improve the efficiency of training deep neural networks. By identifying these winning tickets and training only the sparse subnetworks, researchers can reduce computational costs associated with training while maintaining or even improving model accuracy. This concept has led to the development of pruning techniques aimed at discovering these winning tickets and accelerating the training process.

## NEq

### Neuronal equilibrium

The concept of neuronal equilibrium aims to detect when a neuron reaches a state of equilibrium, indicating that it has learned a particular input-output mapping. The idea is to understand when the neuron has reach a configuration in which he does not require further updates.

To assess this we can evaluate  cosine similarity between all the outputs of the $i$-th neuron at time $t$ and at time $t-1$ for the whole validation set $\Xi_{val}$ as 

\begin{equation}
    \phi_{i}^t = \sum_{\xi\in \Xi_{val}} \sum_{n=1}^{N_i} \hat{y}_{i,n,\xi}^{t} \cdot \hat{y}_{i,n,\xi}^{t-1}.
\end{equation}

The neuron $i$-th reaches the equilibrium when $(\phi_{i})_t$ stops evolving. In this sense to know when the neuron has reached the equilibrium  we need to detect when :

\begin{equation}
    \lim_{t\rightarrow \infty} \phi_{i}^t = k,
\end{equation}


Since it is not trivial to assess this statment we prefer to work with variations of $\phi_{i}_t$ that can be defined as : 
\begin{equation}
    v_{\Delta \phi_i}^t = \Delta \phi_i^t - \mu_{eq} v_{\Delta \phi_i}^{t-1},
\end{equation}

With $\mu_{eq}$ the momentum coefficient.

This only lead to a reformulation of the problem has the equilibrium is reached when we have : $\Delta \phi_i^t \rightarrow 0$

Since we want to track the evolution of $\Delta \phi_i^t$ over time we introduce the velocity of the variations:

\begin{equation}
    v_{\Delta \phi_i}^t = \Delta \phi_i^t - \mu_{eq} v_{\Delta \phi_i}^{t-1},
\end{equation}

With $\mu_{eq}$ the momentum coefficient.

Rewrited :

\begin{equation}
    v_{\Delta \phi_i}^t = \left\{
    \begin{array}{ll}
        \phi_i^{t} + \sum\limits_{m=1}^t (-1)^m \left[(\mu_{eq})^{m-1}+(\mu_{eq})^m\right] \phi_i^{t-m} & \mu_{eq} \neq 0\\
        \phi_i^{t} - \phi_i^{t-1} & \mu_{eq} = 0,
    \end{array}
    \right .
\end{equation}

We need to have $$\mu_{eq} \in [0; 0.5]$$ to prevent the velocity from exploding.

Finally we can set the condition for the neuron to be at the equilibrium as:
\begin{equation}
    \left| v_{\Delta \phi}^t \right | < \varepsilon,~~~~~\varepsilon \geq 0.
\end{equation}

It is important to know that this relation might not hold for all $t$ since there coulb be an instant $t' < t$ where the relation does not hold anymore and the neuron is attracted to a new state and need to be updated again.

### Training scheme

The training scheme can be presented according to this scheme:



![creusilet/winter](http://localhost:1313/images_Winter_Creusillet/prunedbackprop-scheme_full-1.png)

At the first epoch each neuron is considered to be at non-equilibrium. After the first epoch the training scheme can be described as followed:

- An epoch of training is made for all trainable neurons on the training set.
- The training either stops due to the end of training criterion being met or continues to the next step
- The velocity of the similarities are evaluated for every neuron
- The set of trainable neron is determined for the next step accorrding to the equilibrium criterion.

Comparing with regular training we can see two more hyper-parameters:

- $\epsilon$ which determines the threshold at which a neuron is considered to be at equilibrium according to the velocity of the similarities.
- $\mu_{eq}$ which intervenes into the calculation of the velocity of the similarities.
  
## Experiments

### SGD vs Adam

![adam/sgd](http://localhost:1313/images_Winter_Creusillet/adam-vs-sgd-epochs-adam-1.png)
![adam/sgd](http://localhost:1313/images_Winter_Creusillet/adam-vs-sgd-epochs-sgd-1.png)


The authors conduct an experiment comparing two training methods for a ResNet-32 neural network on the CIFAR-10 dataset. The methods compared are SGD (Stochastic Gradient Descent) with momentum and Adam, which are both optimization algorithms used to update network weights iteratively.

In the experiment, the authors observe the FLOPs required for a back-propagation step and the number of updated neurons during training. They note that at high learning rates, more neurons are trained and more FLOPs are required. This is attributed to the network not being at equilibrium—essentially, the network parameters are still very fluid and subject to change, thus requiring more computation.

As training progresses and the learning rate is reduced, fewer neurons need updating, as the network moves towards its final, more stable configuration. The authors find that Adam brings the network towards this equilibrium faster than SGD, but also note that in this specific task, SGD achieves a slightly higher final accuracy than Adam. This may suggest that while Adam is efficient in reaching a state where few neuron weights are updated, SGD's ability to explore the solution space more thoroughly leads to a better generalization on the test data.

The experiment also highlights an interesting behavior at the first learning rate decay around epoch 100 for SGD. The number of updated neurons decreases and then increases, which is not observed with Adam. This difference illustrates the contrasting approaches of the two optimizers: SGD, by reducing the learning rate, encourages continued exploration, which temporarily stabilizes the network until it adjusts to the new learning rate and begins exploring again. Adam, with its adaptive learning rate for each parameter, does not exhibit this behavior because it consistently steers the network towards a stable state.

### Distribution of $\phi$ & choice of $µ_{eq}$

![creusilet/winter](http://localhost:1313/images_Winter_Creusillet/mu-line-1.png)



The paper also discusses the distribution of $\phi$ and the choice of a parameter called $µ_{eq}$ during the training of neural networks.

The parameter $\phi$ measures the cosine similarity between the outputs of a particular neuron at two consecutive training epochs, over the validation set. It is used to determine if a neuron's output has reached equilibrium, meaning its outputs do not significantly change over successive epochs. If $\phi$ equals 1, it indicates that the neuron's output is stable across the epochs, signifying it has reached equilibrium.

The paper further discusses the dynamics of neurons as they approach equilibrium. To quantify this, they introduce a metric called ∆φ, which is the difference in the $\phi$ values across epochs, and $v_{∆\phi}$, which measures the velocity of this change considering a momentum coefficient $µ_{eq}$. This coefficient is important as it determines how much previous changes impact the current measurement of the equilibrium state.

By examining different values for $µ_{eq}$, the paper finds that setting $µ_{eq}$ to 0.5 provides a good compromise, as it ensures a balance between memory of past variations and responsiveness to new changes. This finding is illustrated in the paper's Figure 5, which shows the distribution of $\phi$, $∆\phi$, and $v_{∆\phi}$ for a ResNet-32 model trained on CIFAR-10.

In summary, the authors find that a neuron is at equilibrium if the velocity of the similarity changes, considering the momentum, is below a certain threshold. They also observe that during training, even after reaching equilibrium, neurons may occasionally "unfreeze" and require updates if the learning dynamics change, for instance, if the learning rate is adjusted

### Impact of the validation set size and ε

the authors found that the size of the validation set does not significantly impact the performance of the model. Interestingly, even with a validation set as small as a single image, the method yields good results. This is attributed to the presence of convolutional layers in the network, which, even with a small number of images, generate high-dimensional outputs in each neuron. Additionally, the homogeneity of the dataset (CIFAR-10) likely contributes to the robustness of the performance against changes in the validation set size.

When examining the impact of the parameter ε, which is used to determine when a neuron is at equilibrium and hence does not need to be updated, the authors observe a drop in model performance at very high values of ε. They suggest a value of 0.001 as a good compromise for classification tasks, striking a balance between model performance and computational efficiency.


<style
TYPE="text/css">

code.has-jax {font:
inherit;
font-size:
100%;
background:
inherit;
border:
inherit;}

</style>





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
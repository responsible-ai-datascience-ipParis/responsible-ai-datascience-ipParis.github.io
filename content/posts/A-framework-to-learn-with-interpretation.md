+++
title = 'A Framework to Learn With Interpretation'
date = 2024-02-13T16:56:04+01:00
draft = false
+++

<hr></hr>
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

<script
type="text/x-mathjax-config">

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

<script
type="text/javascript"
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# 1 Introduction
In recent years, the field of machine learning has witnessed a surge in the adoption of complex predictive models across various domains such as law, healthcare, and defense. With the increasing complexity of these models, the need for interpretability has become paramount to ensure trustworthiness, transparency, and accountability. Interpretability, often interchangeably used with explainability, refers to the ability of a model to provide human-understandable insights into its decision-making process. However, it is essential to distinguish between the two terms: interpretability focuses on providing insights into the decision process, while explainability involves logical explanations or causal reasoning, which often necessitate more sophisticated frameworks.

Addressing the challenge of interpreting models, especially deep neural networks, has led to the development of two main approaches: post-hoc methods and "by design" methods. Post-hoc approaches analyze pre-trained systems locally to interpret their decisions, while "interpretable by design" methods aim to integrate interpretability directly into the learning process. Each approach has its advantages and drawbacks, with post-hoc methods being criticized for computational costs and robustness issues, and interpretable systems by design facing the challenge of maintaining performance.

Taking a novel perspective on learning interpretable models, a new generic task in machine learning called Supervised Learning with Interpretation (SLI) is introduced. SLI involves jointly learning a pair of dedicated models: a predictive model and an interpreter model, to provide both interpretability and prediction accuracy. This approach acknowledges that prediction and interpretation are distinct but closely related tasks, each with its own criteria for assessment and hypothesis space. This leads to the introduction of FLINT (Framework to Learn With INTerpretation), a solution to SLI specifically designed for deep neural network classifiers.
<br><br>
FLINT's Key Contributions:

- FLINT presents an original interpreter network architecture based on hidden layers of the network, enabling local and global interpretability through the extraction of high-level attribute functions.
  
- A novel criterion based on entropy and sparsity is proposed to promote conciseness and diversity in the learnt attribute functions, enhancing interpretability.
  
- FLINT can be specialized for post-hoc interpretability, further extending its applicability and demonstrating promising results, as detailed in supplementary materials.

In this blog post, we delve into the significance of interpretability in machine learning systems, exploring its implications, challenges, and recent advancements introduced through FLINT. We will dissect the key concepts presented in the article and examine how FLINT addresses the pressing need for interpretable models, particularly in the context of deep neural networks. Additionally, we will discuss the potential impact of FLINT on various real-world applications and its implications for the future of transparent and trustworthy AI systems. Stay tuned for a comprehensive analysis of FLINT and its contributions to the evolving landscape of interpretable machine learning.

# 2 Learning a classifier and its interpreter with FLINT

The text introduces Supervised Learning with Interpretation (SLI), a new task aimed at incorporating interpretability alongside prediction in machine learning models. In SLI, a separate model, called an interpreter, is employed to interpret the predictions made by the primary predictive model. The task involves minimizing a combined loss function consisting of prediction error and interpretability objectives. The paper focuses on addressing SLI within the context of deep neural networks for multi-class classification tasks. It proposes a framework called Framework to Learn with INTerpretation (FLINT), which utilizes a specialized architecture for the interpreter model, distinguishes between local and global interpretations, and introduces corresponding penalties in the loss function to achieve the desired interpretability.<br>
So for a dataset $S$ and a given model $f \in F$ where $F$ is a class of classifiers (here neural networks) and an interpreter model $g \in G_f$ where $G_f$ is a family of models the SLI problem is presented by:
$$
\arg{\min_{f \in F, g \in G_f}{L_{pred}(f, S) + L_{int}(f, g, S)}}
$$
Where $L_{pred}(f, S)$ denotes a loss term related to prediction error and $L_{int}(f, g, S)$ measures the ability of $g$ to provide interpretations of predictions by $f$.

## 2.1 design of FLINT

![design of FLINT](/images/FLINT_design.png)

<!-- As can be seen in the image, in FLINT we have a prediction model $f$ and an interpreter model $g$. The input of FLINT is a vector $x \in X$ with $X=R^d$ here and the output is a vector $y \in Y$ where $Y = \{y \in \{0, 1\}^C, \sum_{j=1}^C y^j=1 \}$ and $C$ is the number of classes. The prediction model $f$ is a deep neural network with $l$ hidden layers such that $f = f_{l+1} \circ f_l \circ ... \circ f_1$. We have $f_k$ is a hidden layer with $f_k: R^{d_{k-1}} \rightarrow R^{d_k}$. To interpret the outputs of $f$ we randomly select a random subset of $T$ hidden layers of indexes in $I=\{i_1, i_2, ..., i_T\}$ and feed them to the model $g$. Before feeding the output of these layers to $g$ we concetenate them to form a new vector $f_I (x) \in R^D$ with $D = \sum_{t=1}^T d_{i_t}$. The vector $f_I (x)$ is fed to a neural network $\Psi$ to give an output vector $\Phi(x) = \Psi(f_I(x)) \in R^J$ which is an attribute dictionnary composed of functions $\phi_j: X \rightarrow R^+, j=1...J$ whose non negative images $\phi_j (x)$ can be interpreted as the activation of some high level attribute of a "concept" over $X$. Finally, $g$ computes the composition of the attribute dictionnary with an interpretable function $h: R^J \rightarrow Y$.
$$
\forall x \in X, g(x) = h(\Phi(x))
$$
For now we take $h(x) = softmax(W^T \Phi(x))$ but $h$ can be any interpretable function (like a decsion tree for example).
 -->

In FLINT, depicted in the image, we utilize both a prediction model ($f$) and an interpreter model ($g$). The input to FLINT is a vector $x \in X$, where $X = \mathbb{R}^d$, and the output is a vector $y \in Y$, where $Y$ is defined as the set of of one-hot encoding vectors with binary components of size $C$ (the number of classes to predict). The prediction model $f$ is structured as a deep neural network with $l$ hidden layers, represented as $f = f_{l+1} \circ f_l \circ \ldots \circ f_1$. Each $f_k$ represents a hidden layer mapping from $R^{d_{k-1}}$ to $R^{d_k}$. To interpret the outputs of $f$, we randomly select a subset of $T$ hidden layers, indexed by $I=\\{i_1, i_2, \ldots, i_T\\}$, and concatenate their outputs to form a new vector $f_I(x) \in \mathbb{R}^D$, where $D = \sum_{t=1}^T d_{i_t}$. This vector is then fed into a neural network $\Psi$ to produce an output vector $\Phi(x) = \Psi(f_I(x)) \in \mathbb{R}^J$, representing an attribute dictionary comprising functions $\phi_j: X \rightarrow \mathbb{R}^+$, where $\phi_j(x)$ captures the activation of a high-level attributes or a "concept" over $X$. Finally, $g$ computes the composition of the attribute dictionnary with an interpretable function $h: R^J \rightarrow Y$.
$$
\forall x \in X, g(x) = h(\Phi(x))
$$
For now we take $h(x) = softmax(W^T \Phi(x))$ but $h$ can be any interpretable function (like a decsion tree for example).

## 2.2 Interpretation in FLINT

<!-- The interpreter being defined, we need to specify its expected role and corresponding interpretability objective. In FLINT, interpretation is seen as an additional task besides prediction. We are interested by two kinds of interpretation, one at the global level that helps to understand which attribute functions are useful to predict a class and the other at the local level, that indicates which attribute functions are involved in prediction of a specific sample. As a preamble, note that, to interpret a local prediction $f(x)$, we require that the interpreter output $g(x)$ matches $f(x)$. When the two models disagree, we provide a way to analyze the conflictual data and possibly raise an issue about the confidence on the prediction $f(x)$. To define local and global interpretation, we rely on the notion of relevance of an attribute. Given an interpreter with parameter $\Theta_g = (\theta_\Psi, \theta_h)$ and some input $x$, the relevance score of an attribute $\phi_j$ is defined regarding the prediction $g(x) = f(x) = \hat{y}$. Denoting $\hat{y} \in Y$ the index of the predicted class and $w_{j,\hat{y}} \in W$ the coefficient associated to this class, the contribution of attribute $\phi_j$ to unnormalized score of class $\hat{y}$ is $\alpha_{j, \hat{y}, x} = \phi_j(x).w_{j, \hat{y}}$. The relevance score is computed by normalizing contribution $\alpha$ as $r_{j, x} = \frac{\alpha_{j, \hat{y}, x}}{\max_i \lvert \alpha_{i, \hat{y}, x  \lvert}}$ . An attribute $\phi_j$ is considered as relevant for a local prediction if it is both activated and effectively used in the linear (logistic) model. The notion of relevance of an attribute for a sample is extended to its "overall" importance in the prediction of any class $c$. This can be done by simply averaging relevance scores from local interpretations over a random subset or whole of the training set $S$, where predicted class is $c$. Thus, we have: $r_{j, c} = \frac{1}{\lvert S_c \lvert} \sum_{x \in S_c} r_{j, x}$, $S_c = \\{x \in S \lvert \hat{y} = c \\}$. Now, we can introduce the notions of local and global interpretations that the interpreter will provide. -->

With the interpreter defined, let's clarify its role and interpretability objectives within FLINT. Interpretation serves as an additional task alongside prediction. We're interested in two types: global interpretation, which aids in understanding which attribute functions contribute to predicting a class, and local interpretation, which pinpoints the attribute functions involved in predicting a specific sample.

To interpret a local prediction $f(x)$, it's crucial that the interpreter's output $g(x)$ aligns with $f(x)$. Any discrepancy prompts analysis of conflicting data, potentially raising concerns about the prediction's confidence. 

To establish local and global interpretation, we rely on attribute relevance. Given an interpreter with parameters $\Theta_g = (\theta_\Psi, \theta_h)$ and an input $x$, an attribute $\phi_j$'s relevance is defined concerning the prediction $g(x) = f(x) = \hat{y}$. The attribute's contribution to the unnormalized score of class $\hat{y}$ is $\alpha_{j, \hat{y}, x} = \phi_j(x) \cdot w_{j, \hat{y}}$, where $w_{j, \hat{y}}$ is the coefficient associated with this class. Relevance score $r_{j, x}$ is computed by normalizing $\alpha$ as $r_{j, x} = \frac{\alpha_{j, \hat{y}, x}}{\max_i |\alpha_{i, \hat{y}, x}|}$. An attribute $\phi_j$ is considered relevant for a local prediction if it's both activated and effectively used in the linear model.

Attribute relevance extends to its overall importance in predicting any class $c$. This is achieved by averaging relevance scores from local interpretations over a random subset or the entirety of the training set $S$ where the predicted class is $c$. Thus, $r_{j, c} = \frac{1}{|S_c|} \sum_{x \in S_c} r_{j, x}$, where $S_c = \\{x \in S \mid \hat{y} = c\\}$.

Now, let's introduce the local and global interpretations the interpreter will provide:

<!-- Definition 1 (Global and Local Interpretation) For a prediction network $f$, the global interpretation $G(g, f)$ provided by an interpreter $g$, is the set of class-attribute pairs $(c, \phi_j)$ such that their global relevance $r_{j, c}$ is greater than some threshold $1\/\tau$, $\tau > 1$. A local interpretation for a sample $x$ provided by an interpreter $g$ of $f$ denoted $L(x, g, f)$ is the set of attribute functions $\phi_j$ with local relevance score $r_{j, x}$ greater than some threshold $1\/\tau$, $\tau > 1$. It is important to note that these definitions do not prejudge the quality of local and global interpretations. Next, we convert desirable properties of the interpreter into specific loss functions. -->
- Global interpretation ($G(g, f)$) identifies class-attribute pairs $(c, \phi_j)$ where the global relevance $r_{j, c}$ exceeds a threshold $\frac{1}{\tau}$.

- Local interpretation ($L(x, g, f)$) for a sample $x$ includes attribute functions $\phi_j$ with local relevance $r_{j, x}$ surpassing $\frac{1}{\tau}$. These definitions don't assess interpretation quality directly.

## 2.3 Learning by imposing interpretability properties

For learning, we will define certain penalties to minimize, where each one aims to enforce a certain desirable property.

*Fidelity to output:* The output of $g(x)$ should be close to $f(x)$ for any x. This can be imposed through a cross-entropy loss:
$$
L_{of}(f, g, S) = - \sum_{x \in S} h(\Psi(f_I(x)))^T \log(f(x))
$$

<!-- *Conciseness and Diversity of Interpretations:* For any given sample $x$, we wish to get a small
number of attributes in its associated local interpretation. This property of conciseness should make
the interpretation easier to understand due to fewer attributes to be analyzed and promote the "high level"
character in the encoded concepts. However, to encourage better use of available attributes
we also expect activation of multiple attributes across many randomly selected samples. We refer
to this property as diversity. This is also important to avoid the case of attribute functions being
learnt as class exclusive (for eg. reshuffled version of class logits). To enforce these conditions we
utilize notion of entropy defined for real vectors to solve problem of
efficient image search. For a real-valued vector v, the entropy is defined as $\mathcal{E}(v) = - \sum_i p_i \log(p_i)$, $p_i = \exp(v_i)/(\sum_i \exp(v_i))$ Conciseness is promoted by minimizing $\mathcal{E}(\Psi(f_I(x)))$ and diversity is promoted by maximizing
entropy of average 	$\mathcal{E}(f_I(x))$ over a mini-batch. Note that this can be seen as encouraging the
interpreter to find a sparse and diverse coding of $f_I(x)$ using the function. Since entropy-based
losses have inherent normalization, they do not constrain the magnitude of the attribute activation.
This often leads to poor optimization. Thus, we also minimize the $l_1$ norm $\lVert \Psi(f_I(x)) \lVert_1 $ (with
hyperparameter $\eta$) to avoid it. Note that $ l_1 $-regularization is a common tool to encourage sparsity and
thus conciseness, however we show in the experiments that entropy provides a more effective way.
$$
L_{cd}(f, g, S) = -\mathcal{E}(\frac{1}{\lvert S \lvert} \sum_{x \in S} \Psi(f_I(x))) + \sum_{x \in S} \mathcal{E}(\Psi(f_I(x))) + \sum_{x \in S} \eta \lVert \Psi(f_I(x)) \lVert_1
$$ -->

*Conciseness and Diversity of Interpretations:* We aim for concise local interpretations, containing only essential attributes per sample, promoting clearer understanding and capturing high-level concepts. Simultaneously, we seek diverse interpretations across samples to prevent attribute functions from being class-exclusive. To achieve this, we leverage entropy (defined for a vector as $\mathcal{E}(v) = - \sum_i p_i \log(p_i)$), which quantifies uncertainty in real vectors. Conciseness is fostered by minimizing the entropy of the interpreter's output, $\Psi(f_I(x))$, while diversity is encouraged by maximizing the entropy of the average $\Psi(f_I(x))$ over a mini-batch. This approach promotes sparse and varied coding of $f_I(x)$, enhancing interpretability. However, as entropy-based losses lack attribute activation constraints, leading to suboptimal optimization, we also minimize the $l_1$ norm of $\Psi(f_I(x))$ with hyperparameter $\eta$. Although $l_1$-regularization commonly encourages sparsity, our experiments show that entropy-based methods are more effective.
$$
L_{cd}(f, g, S) = -\mathcal{E}(\frac{1}{\lvert S \lvert} \sum_{x \in S} \Psi(f_I(x))) + \sum_{x \in S} \mathcal{E}(\Psi(f_I(x))) + \sum_{x \in S} \eta \lVert \Psi(f_I(x)) \lVert_1
$$

<!-- *Fidelity to input:* To encourage encoding high-level patterns related to input in $\Phi(x)$, we use a decoder network $d : R^J \rightarrow X$ that takes as input the dictionary of attributes $\Psi(f_I(x))$ and reconstructs $x$. -->

*Fidelity to input:* In order to promote the representation of intricate patterns associated with the input within $\Phi(x)$, we employ a decoder network $d : \mathbb{R}^J \rightarrow X$. This network is designed to take the attribute dictionary $\Psi(f_I(x))$ as input and reconstruct the original input $x$.
$$
L_{if}(f, g, d, S) = \sum_{x \in S} (d(\Psi(f_I(x))) - x)^2
$$

Given the proposed loss terms, the loss for the interpretability model writes as follows:
$$
L_{int}(f, g, d, S) = \beta L_{of}(f, g, S) + \gamma L_{if}(f, g, d, S) + \delta L_{cd}(f, g, S)
$$
Where $\beta, \gamma, \delta$ are non-negative hyperparameters. the total loss to be minimized $L = L_{pred} + L_{int}$, where the prediction loss, $L_{pred}$, is the well-know cross entropy loss (since this a classification problem).
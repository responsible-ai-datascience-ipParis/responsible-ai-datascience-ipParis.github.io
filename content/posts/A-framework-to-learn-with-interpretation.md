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

<h1 style="font-size: 36px;">A Framework to Learn With Interpretation</h1>

**Authors: Maroun ABOU BOUTROS, Mohamad EL OSMAN**

# Table of Contents

- [Introduction](#section-1)
- [Learning a classifier and an interpreter](#section-2)
    - [Design of FLINT](#section-2.1)
    - [Interpretation in FLINT](#section-2.2)
    - [Learning by imposing interpretability properties](#section-2.3)
- [Understanding encoded concepts in FLINT](#section-3)
- [Comparison of FLINT to other state-of-the art models](#section-4)
    - [Predictive performance of FLINT](#section-4.1)
    - [Fidelity of Interpreter](#section-4.2)
    - [Conciseness of interpretations](#section-4.3)
- [Reproducing the experiments](#section-5)
    - [Global interpretation](#section-5.1)
    - [Local interpretation](#section-5.2)
- [Subjective evaluation](#section-6)
- [Specialization of FLINT to post-hoc interpretability](#section-7)
- [Final Thoughts](#section-8)

# 1 Introduction {#section-1}

In this post, we’ll explore FLINT, a framework introduced in the paper titled “A framework to learn with interpretation” by Jayneel Parekh, Pavlo Mozharovskyi and Florence d’Alché-Buc, addressing the crucial need for interpretability in machine learning as complex predictive models become more prevalent in fields like law, healthcare, and defense. Interpretability, synonymous with explainability, provides insights into a model’s decision-making process. Two main approaches, post-hoc methods and “interpretable by design” methods, tackle the challenge of interpreting models, each with its pros and cons. A new approach, Supervised Learning with Interpretation (SLI), jointly learns a predictive model and an interpreter model. FLINT, specifically designed for deep neural network classifiers, introduces a novel interpreter network architecture promoting local and global interpretability. It also proposes a criterion for concise and diverse attribute functions, enhancing interpretability. We’ll delve into FLINT’s significance, challenges, and implications for transparent AI systems, and we will reproduce some experiments done in the experimental section of the article and evaluate their outputs to study FLINT's performance.

# 2 Learning a classifier and its interpreter with FLINT {#section-2}

The paper introduces Supervised Learning with Interpretation (SLI), a new task aimed at incorporating interpretability alongside prediction in machine learning models. In SLI, a separate model, called an interpreter, is employed to interpret the predictions made by the primary predictive model. The task involves minimizing a combined loss function consisting of prediction error and interpretability objectives. The paper focuses on addressing SLI within the context of deep neural networks for multi-class classification tasks. It proposes a framework called Framework to Learn with INTerpretation (FLINT), which utilizes a specialized architecture for the interpreter model, distinguishes between local and global interpretations, and introduces corresponding penalties in the loss function to achieve the desired interpretability.<br>
So for a dataset $S$ and a given model $f \in F$ where $F$ is a class of classifiers (here neural networks) and an interpreter model $g \in G_f$ where $G_f$ is a family of models the SLI problem is presented by:
$$
\arg{\min_{f \in F, g \in G_f}{L_{pred}(f, S) + L_{int}(f, g, S)}}
$$
Where $L_{pred}(f, S)$ denotes a loss term related to prediction error and $L_{int}(f, g, S)$ measures the ability of $g$ to provide interpretations of predictions by $f$.

## 2.1 Design of FLINT {#section-2.1}

![design of FLINT](/images/FLINT/FLINT_design.png)

<!-- As can be seen in the image, in FLINT we have a prediction model $f$ and an interpreter model $g$. The input of FLINT is a vector $x \in X$ with $X=R^d$ here and the output is a vector $y \in Y$ where $Y = \{y \in \{0, 1\}^C, \sum_{j=1}^C y^j=1 \}$ and $C$ is the number of classes. The prediction model $f$ is a deep neural network with $l$ hidden layers such that $f = f_{l+1} \circ f_l \circ ... \circ f_1$. We have $f_k$ is a hidden layer with $f_k: R^{d_{k-1}} \rightarrow R^{d_k}$. To interpret the outputs of $f$ we randomly select a random subset of $T$ hidden layers of indexes in $I=\{i_1, i_2, ..., i_T\}$ and feed them to the model $g$. Before feeding the output of these layers to $g$ we concetenate them to form a new vector $f_I (x) \in R^D$ with $D = \sum_{t=1}^T d_{i_t}$. The vector $f_I (x)$ is fed to a neural network $\Psi$ to give an output vector $\Phi(x) = \Psi(f_I(x)) \in R^J$ which is an attribute dictionnary composed of functions $\phi_j: X \rightarrow R^+, j=1...J$ whose non negative images $\phi_j (x)$ can be interpreted as the activation of some high level attribute of a "concept" over $X$. Finally, $g$ computes the composition of the attribute dictionnary with an interpretable function $h: R^J \rightarrow Y$.
$$
\forall x \in X, g(x) = h(\Phi(x))
$$
For now we take $h(x) = softmax(W^T \Phi(x))$ but $h$ can be any interpretable function (like a decsion tree for example).
 -->

In FLINT, depicted in the image, both a prediction model ($f$) and an interpreter model ($g$) are used. The input to FLINT is a vector $x \in X$, where $X = \mathbb{R}^d$, and the output is a vector $y \in Y$, where $Y$ is defined as the set of of one-hot encoding vectors with binary components of size $C$ (the number of classes to predict). The prediction model $f$ is structured as a deep neural network with $l$ hidden layers, represented as $f = f_{l+1} \circ f_l \circ \ldots \circ f_1$. Each $f_k$ represents a hidden layer mapping from $R^{d_{k-1}}$ to $R^{d_k}$. To interpret the outputs of $f$, we randomly select a subset of $T$ hidden layers, indexed by $I=\\{i_1, i_2, \ldots, i_T\\}$, and concatenate their outputs to form a new vector $f_I(x) \in \mathbb{R}^D$, where $D = \sum_{t=1}^T d_{i_t}$. This vector is then fed into a neural network $\Psi$ to produce an output vector $\Phi(x) = \Psi(f_I(x)) = (\phi_1(x), ..., \phi_J (x)) \in \mathbb{R}^J$, representing an attribute dictionary comprising functions $\phi_j: X \rightarrow \mathbb{R}^+$, where $\phi_j(x)$ captures the activation of a high-level attributes or a "concept" over $X$. Finally, $g$ computes the composition of the attribute dictionnary with an interpretable function $h: R^J \rightarrow Y$.
$$
\forall x \in X, g(x) = h(\Phi(x))
$$
For now we take $h(x) = softmax(W^T \Phi(x))$ but $h$ can be any interpretable function (like a decsion tree for example).

## 2.2 Interpretation in FLINT {#section-2.2}

<!-- The interpreter being defined, we need to specify its expected role and corresponding interpretability objective. In FLINT, interpretation is seen as an additional task besides prediction. We are interested by two kinds of interpretation, one at the global level that helps to understand which attribute functions are useful to predict a class and the other at the local level, that indicates which attribute functions are involved in prediction of a specific sample. As a preamble, note that, to interpret a local prediction $f(x)$, we require that the interpreter output $g(x)$ matches $f(x)$. When the two models disagree, we provide a way to analyze the conflictual data and possibly raise an issue about the confidence on the prediction $f(x)$. To define local and global interpretation, we rely on the notion of relevance of an attribute. Given an interpreter with parameter $\Theta_g = (\theta_\Psi, \theta_h)$ and some input $x$, the relevance score of an attribute $\phi_j$ is defined regarding the prediction $g(x) = f(x) = \hat{y}$. Denoting $\hat{y} \in Y$ the index of the predicted class and $w_{j,\hat{y}} \in W$ the coefficient associated to this class, the contribution of attribute $\phi_j$ to unnormalized score of class $\hat{y}$ is $\alpha_{j, \hat{y}, x} = \phi_j(x).w_{j, \hat{y}}$. The relevance score is computed by normalizing contribution $\alpha$ as $r_{j, x} = \frac{\alpha_{j, \hat{y}, x}}{\max_i \lvert \alpha_{i, \hat{y}, x  \lvert}}$ . An attribute $\phi_j$ is considered as relevant for a local prediction if it is both activated and effectively used in the linear (logistic) model. The notion of relevance of an attribute for a sample is extended to its "overall" importance in the prediction of any class $c$. This can be done by simply averaging relevance scores from local interpretations over a random subset or whole of the training set $S$, where predicted class is $c$. Thus, we have: $r_{j, c} = \frac{1}{\lvert S_c \lvert} \sum_{x \in S_c} r_{j, x}$, $S_c = \\{x \in S \lvert \hat{y} = c \\}$. Now, we can introduce the notions of local and global interpretations that the interpreter will provide. -->

With the interpreter defined, let's clarify its role and interpretability objectives within FLINT. Interpretation serves as an additional task alongside prediction. We're interested in two types: global interpretation, which aids in understanding which attribute functions contribute to predicting a class, and local interpretation, which pinpoints the attribute functions involved in predicting a specific sample.

To interpret a local prediction $f(x)$, it's crucial that the interpreter's output $g(x)$ aligns with $f(x)$. Any discrepancy prompts analysis of conflicting data, potentially raising concerns about the prediction's confidence. 

To establish local and global interpretation, we rely on attribute relevance. Given an interpreter with parameters $\Theta_g = (\theta_\Psi, \theta_h)$ and an input $x$, an attribute $\phi_j$'s relevance is defined concerning the prediction $g(x) = f(x) = \hat{y}$. The attribute's contribution to the unnormalized score of class $\hat{y}$ is $\alpha_{j, \hat{y}, x} = \phi_j(x) \cdot w_{j, \hat{y}}$, where $w_{j, \hat{y}}$ is the coefficient associated with this class. Relevance score $r_{j, x}$ is computed by normalizing $\alpha$ as $r_{j, x} = \frac{\alpha_{j, \hat{y}, x}}{\max_i |\alpha_{i, \hat{y}, x}|}$. An attribute $\phi_j$ is considered relevant for a local prediction if it's both activated and effectively used in the linear model.

Attribute relevance extends to its overall importance in predicting any class $c$. This is achieved by averaging relevance scores from local interpretations over a random subset or the entirety of the training set $S$ where the predicted class is $c$. Thus, $r_{j, c} = \frac{1}{|S_c|} \sum_{x \in S_c} r_{j, x}$, where $S_c = \\{x \in S \mid \hat{y} = c\\}$.

Now, let's introduce the local and global interpretations the interpreter will provide:

<!-- Definition 1 (Global and Local Interpretation) For a prediction network $f$, the global interpretation $G(g, f)$ provided by an interpreter $g$, is the set of class-attribute pairs $(c, \phi_j)$ such that their global relevance $r_{j, c}$ is greater than some threshold $1\/\tau$, $\tau > 1$. A local interpretation for a sample $x$ provided by an interpreter $g$ of $f$ denoted $L(x, g, f)$ is the set of attribute functions $\phi_j$ with local relevance score $r_{j, x}$ greater than some threshold $1\/\tau$, $\tau > 1$. It is important to note that these definitions do not prejudge the quality of local and global interpretations. Next, we convert desirable properties of the interpreter into specific loss functions. -->
- Global interpretation ($G(g, f)$) identifies class-attribute pairs $(c, \phi_j)$ where the global relevance $r_{j, c}$ exceeds a threshold $\frac{1}{\tau}$.

- Local interpretation ($L(x, g, f)$) for a sample $x$ includes attribute functions $\phi_j$ with local relevance $r_{j, x}$ surpassing $\frac{1}{\tau}$. These definitions don't assess interpretation quality directly.

## 2.3 Learning by imposing interpretability properties {#section-2.3}

For learning, the paper defines certain penalties to minimize, where each one aims to enforce a certain desirable property.

*Fidelity to output:* The output of $g(x)=h(\Psi(f_I(x)))$ should be close to $f(x)$ for any x. This can be imposed through a cross-entropy loss:
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

*Conciseness and Diversity of Interpretations:* We aim for concise local interpretations, containing only essential attributes per sample, promoting clearer understanding and capturing high-level concepts. Simultaneously, we seek diverse interpretations across samples to prevent attribute functions from being class-exclusive. To achieve this, the paper proposes that we leverage entropy (defined for a vector as $\mathcal{E}(v) = - \sum_i p_i \log(p_i)$), which quantifies uncertainty in real vectors. Conciseness is fostered by minimizing the entropy of the interpreter's output, $\Phi(x) = \Psi(f_I(x))$, while diversity is encouraged by maximizing the entropy of the average $\Psi(f_I(x))$ over a mini-batch. This approach promotes sparse and varied coding of $f_I(x)$, enhancing interpretability. However, as entropy-based losses lack attribute activation constraints, leading to suboptimal optimization, we also minimize the $l_1$ norm of $\Psi(f_I(x))$ with hyperparameter $\eta$. Although $l_1$-regularization commonly encourages sparsity, the experiments done show that entropy-based methods are more effective.
$$
L_{cd}(f, g, S) = -\mathcal{E}(\frac{1}{\lvert S \lvert} \sum_{x \in S} \Psi(f_I(x))) + \sum_{x \in S} \mathcal{E}(\Psi(f_I(x))) + \sum_{x \in S} \eta \lVert \Psi(f_I(x)) \lVert_1
$$

<!-- *Fidelity to input:* To encourage encoding high-level patterns related to input in $\Phi(x)$, we use a decoder network $d : R^J \rightarrow X$ that takes as input the dictionary of attributes $\Psi(f_I(x))$ and reconstructs $x$. -->

*Fidelity to input:* In order to promote the representation of intricate patterns associated with the input within $\Phi(x)$, a decoder network $d : \mathbb{R}^J \rightarrow X$ is employed. This network is designed to take the attribute dictionary $\Phi(x)=\Psi(f_I(x))$ as input and reconstruct the original input $x$.
$$
L_{if}(f, g, d, S) = \sum_{x \in S} (d(\Psi(f_I(x))) - x)^2
$$

Given the proposed loss terms, the loss for the interpretability model writes as follows:
$$
L_{int}(f, g, d, S) = \beta L_{of}(f, g, S) + \gamma L_{if}(f, g, d, S) + \delta L_{cd}(f, g, S)
$$
Where $\beta, \gamma, \delta$ are non-negative hyperparameters. the total loss to be minimized $L = L_{pred} + L_{int}$, where the prediction loss, $L_{pred}$, is the well-know cross entropy loss (since this a classification problem).

# 3 Understanding encoded concepts in FLINT {#section-3}

Once the predictor and interpreter networks are jointly learned, interpretation can be conducted at both global and local levels . A critical aspect highlighted by the authors is understanding the concepts encoded by each individual attribute function ​$\phi_j$ . Focusing on image classification, the authors propose representing an encoded concept as a collection of visual patterns in the input space that strongly activate $\phi_j$ . They present a pipeline for generating visualizations for both global and local interpretation, adapting various existing tools .

For global interpretation visualization, the authors propose starting by selecting a small subset of training samples from a given class c that maximally activate ​$\phi_j$ . This subset, referred to as Maximum Activating Samples (MAS), is denoted as $MAS(c , ​\phi_j , l)$ where l is the subset size (set as 3 in their experiments). However, while MAS provides some insight into the encoded concept, further analysis is required to understand the specific aspects of these samples that cause ​$\phi_j$ activation. To achieve this, the authors propose utilizing a modified version of activation maximization called Activation Maximization with Partial Initialization (AM+PI). This technique aims to synthesize input that maximally activates ​$\phi_j$ by optimizing a common activation maximization objective, initialized with a low-intensity version of the sample from MAS. 

For local analysis, given any test sample $x_{0}$ , its local interpretation $L(x_{0},f,g)$ can be determined, representing the relevant attribute functions . To visualize a relevant attribute ​$\phi_j$, the authors suggest repeating the AM+PI procedure with initialization using a low-intensity version of $x_{0}$ to enhance the concept detected by ​$\phi_j$ in $x_{0}$ .

# 4 Comparison of FLINT to other state-of-the art models {#section-4}

The paper delves into the selection of datasets and neural network architectures for their experimental setup, considering four datasets: MNIST, FashionMNIST, CIFAR-10, and a subset of the QuickDraw dataset. For the predictor network f, two architectures are experimented with: a LeNet-based network for MNIST and FashionMNIST, and a ResNet18-based network for QuickDraw and CIFAR. Specific layers are chosen from the last few convolutional layers to effectively capture higher-level features. The number of attributes J is tailored for each dataset: J=25 for MNIST and FashionMNIST, J=24 for the QuickDraw subset, and J=36 for CIFAR. In the paper, the authors thoroughly evaluate and compare their model with other state-of-the-art systems, focusing on accuracy and interpretability. They employ evaluation metrics designed to assess the effectiveness of previously proposed losses for interpretability. Their primary method for comparison, when applicable, is SENN, chosen for its inherent interpretability aligning with FLINT. Additionally, they include PrototypeDNN as a baseline for comparing predictive performance, and LIME and VIBI for evaluating the fidelity of interpretations. 

### 4.1 Predictive performance of FLINT {#section-4.1}

In the article, the authors aim to validate two key aspects related to the predictive performance of FLINT. Firstly, they investigate whether jointly training the predictor $f$ with the interpreter $g$ and backpropagating the loss term $L_{\text{int}}$ adversely affects performance. Secondly, we seek to determine if the achieved performance is comparable to other similarly interpretable models designed from the outset for interpretability.

To address the first goal, they compare the accuracy of the predictor trained with FLINT (denoted as FLINT-$f$) with that of the same predictor architecture trained solely with the $L_{\text{pred}}$ loss (denoted as BASE-$f$). For the second goal, they compare the accuracy of FLINT-$f$ with that of SENN and PrototypeDNN , both designed for interpretability from the outset and not relying on input attribution for interpretations.

The accuracies obtained from these comparisons are presented in the provided table below :

![Results for accuracy (in %) and fidelity to FLINT-f on different datasets](/images/FLINT/Accuracy_results.png)

Their findings indicate that training $f$ within FLINT does not lead to any significant loss in accuracy across any dataset. Furthermore, FLINT demonstrates competitive performance with other interpretable models designed from the outset for interpretability.


### 4.2 Fidelity of Interpreter {#section-4.2}

The paper assess the fidelity of the interpreter, which is defined as the proportion of samples where the predictions of a model and its interpreter agree, indicating the same class label . This metric is commonly used to evaluate how well an interpreter approximates a model. To ensure that the interpreter trained with FLINT (referred to as FLINT-$g$) achieves a satisfactory level of agreement with FLINT-$f$, we conduct a benchmark against a state-of-the-art black-box explainer, VIBI , and the traditional method LIME . The results are presented in the above provided table .

FLINT-$g$ consistently demonstrates higher fidelity compared to the benchmarked methods. Despite the inherent difference in methodology, where FLINT-$g$ accesses intermediate layers while the other systems are black-box explainers, the results clearly indicate that FLINT-$g$ exhibits high fidelity to FLINT-$f$. These findings reinforce the effectiveness of FLINT-$g$ in faithfully representing the predictions of FLINT-$f$.

### 4.3 Conciseness of interpretations {#section-4.3}

In the article, the conciseness of interpretations is evaluated by measuring the average number of important attributes present in generated interpretations. This metric assesses the need for analyzing attributes and is computed for a given sample $x$ by counting the number of attributes ​$\phi_j$ with $r_{j,x}$ greater than a threshold $\frac{1}{\tau}$, where $\tau > 1$. By varying the threshold $\frac{1}{\tau}$, the mean conciseness $\text{CNS}_g$ of $g$ over the test data is computed, where lower conciseness indicates a need to analyze fewer attributes on average.

To compare the conciseness of FLINT with SENN across all four datasets, the authors generate conciseness curves. As shown in the provided figure below , FLINT consistently produces interpretations that are more concise compared to SENN. Notably, SENN tends to consider a majority of concepts as relevant even for lower thresholds (higher $\tau$), indicating less concise interpretations.

![Conciseness comparison of FLINT and SENN](/images/FLINT/Conciseness_curves.png)

## 5 Reproducing the experiments {#section-5}

In pursuit of a comprehensive understanding of the model and its operational dynamics across prevalent datasets, we replicated the study by cloning the project from the GitHub repository referenced in the article. Subsequently, we conducted model executions on GPU resources provided by the educational institution. Our experimentation involved the CIFAR10 and QuickDRAW datasets. The instructions to execute the model on the GitHub repository are clear, and it runs flawlessly. Additionally, there is a well-detailed Python notebook containing code for visualizations of the data, such as attribute relevance and local interpretations, all of which we will explore in this section.

### 5.1 Global interpretation {#section-5.1}

In the article, the authors explore global interpretation using a figure similar to the one provided below which was reproduced from the notebook, and which illustrates the generated global relevances $r_{j,c}$ for all class-attribute pairs in the QuickDraw dataset. 

<!-- ![Global class-attribute relevances](/images/FLINT/Global_class_attribute_QuickDRAW.png) -->
<div style="text-align:center;">
    <img src="/images/FLINT/Global_class_attribute_QuickDRAW.png" alt="Image" width="300" height="200">
</div>

Additionally, by running the model on the CIFAR10 and QuickDRAW dataset we got visual outputs representative of class-attribute pair analyses for both datasets. These outputs served as pivotal tools in elucidating interrelations and facilitating comparative assessments between attributes and classes. We present below two figures derived from the resultant class-attribute pair analyses for each of the 2 datasets for sample images different from those shown in the article.


![Class-attribute pair analysis on dataset CIFAR10](/images/FLINT/Class_attribute_pair_CIFAR10.png)
*Caption: Class-attribute pair analysis on dataset CIFAR10*

![Class-attribute pair analysis on dataset QuickDraw](/images/FLINT/Class_attribute_pair_QuickDraw.png)
*Caption: Class-attribute pair analysis on dataset QuickDraw*

We focus on class-attribute pairs with high relevance, showcasing examples in the provided figure above . For each pair, we examine Maximum Activating Samples (MAS) alongside their corresponding Activation Maximization with Partial Initialization (AM+PI) outputs.


MAS analysis alone provides valuable insights into the encoded concept. For instance, on QuickDRAW dataset, attribute $\phi_{16}$  relevant for class 'Banana' activate the curve shape of the banana. However, AM+PI outputs offer deeper insights by elucidating which parts of the input activate an attribute function more clearly. AM+PI outputs are particularly important for attributes relevant to multiple classes. For example , on CIFAR10 dataset , attribute $\phi_{12}$ activates for 'Deer' class , but the specific focus of the attribute remains ambiguous. The outputs of the AM+PI method indicate that attribute $\phi_{12}$ predominantly highlights the area encompassing the legs and the deer horn, characterized as the most prominently enhanced regions.


### 5.2 Local interpretation {#section-5.2}

Similarly to the article, we explored local interpretation through the figure provided below which was generated in the notebook, which showcases visualizations for 2 test samples of the QuickDRAW dataset. Both predictor $f$ and interpreter $g$ accurately predict the true class in all cases. For each case, they highlighted the top 3 relevant attributes to the prediction along with their relevances and corresponding AM+PI outputs.

![Local interpretations for test samples](/images/FLINT/Local_interpretations.jpg) 

Analysis of the AM+PI outputs reveals that attribute functions generally activate for patterns corresponding to the same concept inferred during global analysis. This consistency is evident for attribute functions present in the previous figures. 

## 6 Subjective evaluation {#section-6}

In the article,  a subjective evaluation survey with 20 respondents using the QuickDraw dataset to assess FLINT's interpretability is conducted. The authors selected 10 attributes covering 17 class-attribute pairs and presented visualizations (3 MAS and AM+PI outputs) along with textual descriptions for each attribute to the respondents. They were asked to indicate their level of agreement with the association between the descriptions and the patterns in the visualizations using predefined choices.

Descriptions were manually generated, including 40% incorrect ones to ensure informed responses. Results showed that for correct descriptions, 77.5% of respondents agreed, 10.0% were unsure, and 12.5% disagreed. For incorrect descriptions, 83.7% disagreed, 7.5% were unsure, and 8.8% agreed. These results affirm that the concepts encoded in FLINT's learned attributes are understandable to humans.

# 7 Specialization of FLINT to post-hoc interpretability {#section-7}

FLINT's versatility extends beyond its primary goal of interpretability by design, allowing for specialization in providing post-hoc interpretations when a classifier $\hat{f}$ is already available. This post-hoc interpretation learning falls under the broader scope of Supervised Layer-wise Interpretation (SLI). It entails constructing an interpreter of $\hat{f}$ by minimizing the loss function $L_{\text{int}}(\hat{f}, g, S)$ with respect to $g$, where $S$ denotes the training set.

Experimental validation of this post-hoc capability is performed by interpreting fixed models trained solely for accuracy, for example the discussed BASE-$f$ models from Section 4.1. Even without fine-tuning the internal layers of $\hat{f}$, the system demonstrates the ability to generate high-fidelity and meaningful interpretations. 

# 8 Final Thoughts {#section-8}

In conclusion, FLINT introduces a pioneering framework for training a predictor network alongside its interpreter network, incorporating specialized losses to offer both local and global interpretations based on learned attributes and concepts. However, this approach raises unresolved queries regarding the faithfulness of interpretations to the predictor. The definition of faithfulness in interpreting decision processes has yet to achieve consensus, particularly concerning post-hoc interpretability and discrepancies between the predictor and interpreter models. While generating interpretations from hidden layers of the predictor network enhances faithfulness to some extent, complete fidelity cannot be ensured due to disparities in the final portions of the predictor and interpreter. Nevertheless, if prioritizing faithfulness by design is deemed paramount, FLINT-g can serve as the ultimate decision-making network, consolidating both roles into a single entity.
+++
title = 'Statistical Minimax Rates Under Privacy'
date = 2024-01-31T17:22:02+01:00
draft = false
+++

# Estimating Privacy in Data Science: A Comprehensive Guide

# Table of Contents

- [Incentives](#section-0)
- [Introduction](#section-1)
- [Definition](#section-2)
- [Theory](#section-3)

## Why do we care about privacy ? {#section-0}

$$ test \frac{1}{2} + \pi$$

Imagine, you're quietly at home when the doorbell rings. You open the door and a government official appears: population census. Even though he shows you his official badge and you'd like to help him in the public interest, you find it hard to answer his questions as you go along. Indeed, the first questions about the date of your move are easy and public. On the other hand, when he asks about the number of children, marital status or your salary and what you do with it, you struggle. Not because you don't know the answer, but because you're faced with an ethical dilemma: transparency towards the state versus protection of personal data. In short, his work goes against your privacy. 

This stress has major consequences: as you doubt what could happen to you with this data, but you still want to answer it, you underestimate your answers. On a wider scale, this leads to a suffrage bias and therefore a lack of knowledge of the real situation of your population. Warner [1], the first to tackle this problem from a statistical angle talks of an evasive bias and says: "for reasons of modesty, fear of being thought bigoted, or merely a reluctance to confide secrets to strangers,respondents to surveys might prefer to be able to answer certain questions non-truthfully, or at least without the interviewer knowing their true response"

This situation presented a trusted agent, in that he wasn't trying to harm you directly. Now imagine that you agree to give him your personal data, but that on the way home, this agent of the state is mugged and someone steals his documents. Not only is this an attack on his person, it's also an attack on yours: as the guarantor of your data, it's now at the mercy of the attacker. The problem here is not to have protected yourself against a malicious agent. 

Admittedly, these situations are rare, but with the densification of data, their analogies are omnipresent: cookies on the Internet, cyber-attacks, datacenter crashes...One area for improvement is quite simply to better certify usage by means of cyber protection labels and leads to such a norm to achieve trust:
![Data Privacy2](http://localhost:1313/images/Antoine_Klein/Umbrella.png)

In this blog, we propose to tackle this problem from a completely different angle: how to both enable the agent to take global measures and prevent it and any subsequent malicious agents from being able to re-identify my personal data. We'll also use minimax bounds to answer the question: for a given privacy criterion, what's the loss in terms of estimation? (fundamental trade-offs between privacy and convergence rate)

## Scientific introduction {#section-1}
<a name="test"></a>

Our blog will follow the same plan as the article that inspired it (John C. Duchi [2]),i.e. to show that response randomization achieves optimal convergence in the case of multinomial estimation, and then that this process can be generalized to any nonparametric distribution estimation. To this end, we will introduce the notion of local differential privacy as well as the minimax theory for obtaining optimal limits. All this will shed light on the trade-off between privacy and estimation rates. We will also explain algorithms to implement these optimal strategies. Finally, we will propose some experimental results.

## Some key definitions {#section-2}

Let assume that you want to make private $X_1 , ... , X_n \in X$ random variable and, as the statistician, you only observe $Z_1, . . . , Z_n ∈ Z$. The paper assumes that there exist a markov kernel that links the true ramdom variables and the observed ones as follow: $Q_i(Z_i | X_i = x)$.

The privacy mechanism is to be said non interactive if each $Z_i$ is obtained only conditionnaly on $X_i$ (and not on the others). This represents the fact that the privacy mechanism is memory less. If not, the mechnism is said to be interactive. 

In the following, we will work only with non-interactive privacy mechanism but in the conlusion we will claim that newer studies showed that it is not enough for some larger problems.

$Z_i$ is said to be α-local-differentially private for the original data $X_i$ if $$sup(\frac{Q(Z | X_i = x)}{Q(Z | X_i = x')} | x, x' ∈ X) ≤ exp(α)$$.  

An intuitive way of understanding this definition is to see that the smaller &alpha; is (the more private it is), the more difficult it is to distinguish the distribution of Z conditional on two different X data. 

## Theoretical results {#section-3}

### The case of multinomial estimation

#### Theorem

In this section, we return back to the problem of the private survey. For the statistician view, estimating a survey is estimating the parameter &theta; from the Bernouilli distribution $B(θ)$. 
This problem is a special case of multinomial estimation, where `θ` is now a multidimensional parameter that is amenable to simplex probability. $∆_d := (θ ∈ ℝ_+ | d, θ ≥ 0, ∑θ_j = 1)$.

Theorem : Given α-local-differentially private $Z_i$, there exists some arbitrary constants $C_1$, $C_2$ such that for all $\alpha\in [0,1]$:
$$C_1 min(1, \frac{1}{\sqrt{n\alpha^2}}, \frac{d}{n\alpha^2}) ≤ E[|θ_{hat} - θ|^2] ≤ C_2 min(1, \frac{d}{n\alpha^2})$$ and 
$$C_1 min(1,\frac{1}{\sqrt{n\alpha^2}}) ≤ E[||θ_{hat} - θ||_1] ≤ C_2 min(1,\frac{d}{\sqrt{n\alpha^2}})$$.

Recall from standard statistics: For non private independant $Z_i$ with finite variance, , there exists some arbitrary constants $C_3$ such that:
$$E[|θ_{hat} - θ|^2] ≤ \frac{C_3}{n}$$

In others term, providing α-local-differentially privacy causes a reduction in the effective sample size of a factor $\frac{\alpha^2}{d}$ for best situations. It thus means that the asymptotically rate of convergences remains unchanged which is a really good news !

#### Practical strategies

The paper deals with the 2 standard methods to implement such a strategy that obtains the minimax rates:
- [Randomized responses](#section-10)
- [Laplace Noise (beyond paper)](#section-11)

##### Randomized responses {#section-10}

The intuition of this section is the following : to not allow the statistician to retrieve your personnal data in case of Bernouilli distribution, you toss a coin. If it is heads, you say to him your reel answer, if it is tails, you say the opposite. In his point of view, as he doesn't know what was the result of the coin, he can't distinguish if you tell the true or not but in a large scale, he knows that he will have half correct answer, half lies so that he can retrieve information. 

For the multinomial estimation now, you will generalize this procedure to the multidimensionnal setting. For each coordinate, you will tell to the statistician the reel answer with a certain probability and lies otherwise. More precisely, its leads to : 

$$[Z]_j = x_j \text{ with probability } \frac{e^\frac{\alpha}{2}} {1 + e^\frac{\alpha}{2}}$$
$$[Z]_j = 1 - x_j \text{ with probability } \frac{1}{1 + e^\frac{\alpha}{2}}$$


Such a mechanism achieves α-local-differentially privacy because one can show that :

$$\frac{Q(Z = z | x)}{Q(Z = z | x')} = e^\frac{\alpha}{2}(||z - x||_1 - ||z - x'||_1) \in [e^{-\alpha}, e^\alpha]$$ which is the criteria given above.

With the notation as $1_d=[1, 1, 1, ..., 1]$ corresponds to a d-vector with each coordinate equals 1, we can also show that :

$$E[Z | x] = \frac{e^\frac{\alpha}{2} - 1}{e^\frac{\alpha}{2} + 1} * x + \frac{1}{1 + e^\frac{\alpha}{2}}1_d$$

This leads to the natural moment-estimator : 

$$θ_{hat} = \frac{1}{n} ∑_{i=1}^{n} \frac{Z_i - 1_d}{1 + e^\frac{\alpha}{2}} * \frac{e^\frac{\alpha}{2} + 1}{e^\frac{\alpha}{2} - 1}$$

One can also show that it verifies :

$$E[ ||θ_{hat}- θ||_2] ≤  \frac{d}{n} * \frac{(e^\frac{\alpha}{2} + 1)^2}{(e^\frac{\alpha}{2} - 1)^2} < \frac{C_3}{nα^2}$$ which is the announced result.

##### Laplace Noise (beyond paper) {#section-11}

Instead of saying the truth with some probability, one may think of adding noise to the answer so that the statistician can't retrieve his real answer. This is exactly the mechanism we propose to dive in and which is not covered in the paper.

Definition: A noise is said to be a Laplace noise with parameters (μ, b) if it verifies:  
$$f(x|μ, b) = \frac{1}{2b} * exp(\frac{-|x - μ|}{b})$$

A visualisation for differents parameters is given below. We can see that Laplace distribution is a shaper verson of the gaussian distribution :
![Laplace](http://localhost:1313/images/Antoine_Klein/Laplace.png)

The trick is to use such a noise. Let assume `Xi ∈ [-M,M]` and construct the private mechanism as follow:   
$$Z_i = X_i + \sigma W_i$$ where $W_i$ is drawn from a Laplace noise (0,1).

One can show that :

$$\frac{Q(Z = z | x)}{Q(Z = z | x')} \leq e^{\frac{1}{\sigma} * |x - x'|} \leq e^{\frac{2M}{\sigma}}$$

Thus, with the choice of `σ = 2M/α`, it verifies α-local-differentially privacy. The proposed estimator is the following :  
`Z_hat = X̄ + (2M/α) W̄`.

One can show that it is an unbiaised estimator that achieves the optimal rates:  
`E[Z_hat] = E[X]`  
`V[Z_hat] = V(X)/n + 4M^2/nα^2 * V(W̄) = V(X)/n + 8M^2/nα^2` so that,  
`E[ ||Z_hat- X||_2] ≤ C3/nα^2`.

This is exactly the optimal rates, quite outstanding !

### The case of density estimation

One accurate question that can raise is the following : what about others distribution ? Is privacy more costly in general cases ? What is the trade-off ?

To answer this question, let's precise the problem. 

We want to estimate in a non-paramtric way a 1D-density function `f` belonging to one of theses classes :  
-Hölder Class (β, l): For all `x,y ∈ R and m<= β`, `|f^(m)(x) - f^(m)(y)| <= L |x - y| ^(β-m)`  
-Sobolev Class : `F_β[C] := { f ∈ L^2([0, 1]) | f = ∑_{j=1}^{∞} θ_jϕ_j such that ∑_{j=1}^{∞} j^2β ϕ_j^2 ≤ C^2 }`

In a intuitition way, those two classes express that `f` is smooth enough to admits Lipschitz constant to its derivative so that it doesn't "vary" locally too much.

#### Theorem

##### Without privacy

One can show that without privacy, the minimax rate achievable for estimating a Hölder Class function is:  
`MSE(f_hat - f) <= C1 * n^(-2β/1+2β)` with the estimator  
`f_hat(x) = 1/n * ∑_{i=1}^{n} 1/h * K(x-X_i/h)` with `h = C2 * n^(-1/2β+1)`

In the case of d-multidimensionnal density `f`, the optimal rate is :  
`MSE(f_hat - f) <= C4 * n^(-2β/d+2β)` with the estimator  
`f_hat(x) = 1/n * ∑_{i=1}^{n} 1/h^d * K^d(x-X_i/h)` with `h = C5 * n^(-1/2β+d)`

This illustrates once again the curse of dimensionnality.

##### With privacy

Let assume that `f` bellongs to one of the two classes with  `β` as smoothness parameter.  
Then, the optimal α-local-differentially private optimal rate is :  
`MSE(f_hat - f) <= C1 * (nα^2)^(-2β/(2β+2))`.

One may observe two pessimistic news:  
-The rate is affected by a factor of α^2 as for the multinomial estimation  
-More damageable: the rate is slower in term of `n` unlike the previous problem which make privacy in this case more costly.

##### Practical strategies

Eventhough this rate is pessimistic and proves that privacy comes at a cost, it remains to illustrates how can we achieves this best but not great rate.
For this end, once again, two strategies are possible.

- [Randomized responses](#section-12)
- [Laplace Noise (beyond paper)](#section-13)

##### Randomized responses {#section-12}

This is the strategy illustrated in the paper and consists of sampling for each coordinate according the realisation of a Bernouilli variable with the correct probability as function of `α`.
As it is not the most comprehensive and straightforward method, we prefer to dive in depth into the second one; uncovered in the paper.

##### Laplace Noise (beyond paper) {#section-13}

Let assume that `Xi ∈ [0,M] almost surely`. We note `G_j = [j-1/K, j/K]` the bin of length `1/K`.

We consider the histogramm estimator: 
`f_hat(x) = K/n ∑_{j=1}^{K} ∑_{i=1}^{n} 1_Xi∈G_j * 1_x∈G_j`.

We now construct the private mechanism as follow:  
`Zi = [1_Xi∈G_1 + 2/α * W_1 , ... , 1_Xi∈G_K + 2/α * W_K]`.

In an intuitive way, we add a Laplace noise realisation for each bin of length `1/K`.

This guarantees α-local-differentially privacy as :  
`Q(Z = z | x)/Q(Z = z | x') <= exp(α/2 * ∑_{j=1}^{K} |1_x∈G_j - 1_x'∈G_j |) <= exp(α/2 * 2)`.

This leads to the α-local-differentially private estimator :  
`f_private_hat = f_hat + 2K/nα ∑_{j=1}^{K} W_j`.

The biais is the same as the unprivate case as :  
`E[f_private_hat] = E[f_hat] + 0`.

One may prove that if f bellongs to the β-Hölder Class:  
`Biais(f_private_hat, f) <= C1 * K^(-β)`

Meanwhile, `V[f_private_hat] <= C2/n + 4*K^2/α^2 * V[W]/n`, such that in total  :
`MSE(f_private_hat - f) <= C1 * K^(-2β) + C2/n + C3*K^2/nα^2`.  
Minimizing over K (hyperparameters) leads to :  
`K = C4 * (nα^2)^(-/2β+2)` and thus to: 
`MSE(f_private_hat - f) <= C5 * (nα^2)^(-2β/2β+2)`, which is the expected bound.

---

## Experiment: Estimating Privacy Using Differential Privacy

### Overview {#section-111}
<a href="#test" style="background-color: yellow; padding: 2px 5px; border-radius: 3px;">Go to test</a>
This is the introduction section. experiment, we focus on utilizing the concept of differential privacy to estimate the privacy level of a dataset. Differential privacy offers a rigorous framework for quantifying the impact of an individual's data on the overall privacy of a dataset, providing a mechanism to balance data utility and privacy protection.

### Methodology

1. **Data Preparation**: We begin by preprocessing the dataset to ensure it adheres to the requirements of differential privacy. This may involve techniques such as data anonymization or perturbation.

2. **Privacy Metric Calculation**: Next, we compute the privacy metric using differential privacy algorithms. This metric quantifies the level of privacy protection afforded to individuals within the dataset.

3. **Evaluation**: We evaluate the effectiveness of our privacy estimation by comparing the computed privacy metric against established thresholds or benchmarks. This step provides insights into the adequacy of privacy protection measures employed.

### Results

Our experiment yielded promising results, demonstrating the feasibility of using differential privacy for privacy estimation in data science. The calculated privacy metric indicated a high level of privacy protection, exceeding industry standards in several instances. However, further analysis is warranted to explore the robustness of our approach across diverse datasets and scenarios.

<script>
function highlight(text) {
  var inputText = document.getElementById("markdown-content");
  var innerHTML = inputText.innerHTML;
  var index = innerHTML.indexOf(text);
  if (index >= 0) { 
    innerHTML = innerHTML.substring(0,index) + "<span class='highlight'>" + innerHTML.substring(index,index+text.length) + "</span>" + innerHTML.substring(index + text.length);
    inputText.innerHTML = innerHTML;
  }
}
highlight("Estimating Privacy in Data Science");

</script>

[Read more about our experiment methodology](##Experiment:-Estimating-Privacy-Using-Differential-Privacy)

### Conclusion

We have linked minimax analysis from statistical decision theory with differential privacy, bringing some of their respective foundational principles into close contact. In this paper particularly, we showed how to apply our divergence bounds to obtain sharp bounds on the convergence rate for certain nonparametric problems in addition to standard finite-dimensional settings. By providing sharp convergence rates for many standard statistical inference procedures under local differential privacy, we have developed and explored some tools that may be used to better understand privacy-preserving statistical inference and estimation procedures. We have identified a fundamental continuum along which privacy may be traded for utility in the form of accurate statistical estimates, providing a way to adjust statistical procedures to meet the privacy or utility needs of the statistician and the population being sampled. Formally identifying this trade-off in other statistical problems should allow us to better understand the costs and benefits of privacy; we believe we have laid some of the groundwork to do so.

# Interactive Markdown Example

This is an interactive Markdown file.

<input type="text" id="inputField" placeholder="Type something...">
<button onclick="displayInput()">Submit</button>
<div id="output"></div>

<script>
    function displayInput() {
        var inputValue = document.getElementById("inputField").value;
        document.getElementById("output").innerText = "You typed: " + inputValue;
    }
</script>

This is <span style="background-color: yellow;">highlighted text</span> using inline CSS.

<span class="highlight-on-hover">Hover over this text to see it highlighted.</span>

# Quiz or Survey Form

Please fill out the quiz form below:

<form id="quiz-form" class="quiz-form">
    <div class="quiz-question">
        <p>What is privacy?</p>
        <div class="quiz-options">
            <label>
                <input type="radio" name="question1" value="1">
                Avoid asking questions that can raise private information
            </label>
            <label>
                <input type="radio" name="question1" value="2">
                A mechanism that prevents other agent to retrieve personnal information in your answer
            </label>
            <label>
                <input type="radio" name="question1" value="3">
                An ethical-washing trend
            </label>
        </div>
        <p>Which situation is α-local-differentially privacy?</p>
        <div class="quiz-options">
            <label>
                <input type="radio" name="question2" value="1">
                sup {Q(Z | Xi = x)/Q(Z | Xi = x')} | x, x' ∈ X} >= exp(α)
            </label>
            <label>
                <input type="radio" name="question2" value="2">
                You tell the truth half the time, you lie otherwise.
            </label>
            <label>
                <input type="radio" name="question2" value="3">
                Z_i = X_i + (2M/α) W_i with W_i drawn from a Laplace Noise(0,1)
            </label>
        </div>
        <p>What is the privacy cost in term of optimal rate ?</p>
        <div class="quiz-options">
            <label>
                <input type="radio" name="question3" value="1">
                Multinomial estimation: A factor α^2/d 
            </label>
            <label>
                <input type="radio" name="question3" value="2">
                Density estimation: from n^(-2β/2β+2) (without privacy) to (nα^2)^(-2β/(2β+2))
            </label>
            <label>
                <input type="radio" name="question3" value="3">
                We loose nothing, that's the surprising finding of the paper
            </label>
        </div>
    </div>
    <!-- Add more quiz questions as needed -->
    <button type="submit" class="quiz-submit">Submit</button>
</form>

<div id="quiz-results" class="quiz-results"></div>

<script>
    // Define quiz questions and correct answers
    const quizQuestions = [
        {
            question: "What is privacy?",
            answer: "2"
        },
        //Add more quiz questions as needed
        {
            question: "Which situation is α-local-differentially privacy?",
            answer: "3"
        },
        //Add more quiz questions as needed
        {
            question: "What is the privacy cost in term of optimal rate ?",
            answer: "1"
        }
    ];

    // Handle form submission
    document.getElementById('quiz-form').addEventListener('submit', function(event) {
        event.preventDefault();

        // Calculate quiz score
        let score = 0;
        quizQuestions.forEach(question => {
            const selectedAnswer = document.querySelector(`input[name="question${quizQuestions.indexOf(question) + 1}"]:checked`);
            if (selectedAnswer) {
                if (selectedAnswer.value.toLowerCase() === question.answer) {
                    score++;
                    selectedAnswer.parentElement.classList.add('correct');
                } else {
                    selectedAnswer.parentElement.classList.add('incorrect');
                }
            }
        });

        // Display quiz results
        const quizResults = document.getElementById('quiz-results');
        quizResults.innerHTML = `<p>You scored ${score} out of ${quizQuestions.length}.</p>`;
    });
</script>

---

By leveraging the principles of differential privacy, data scientists can gain valuable insights into the privacy implications of their analyses and foster a culture of responsible data stewardship. Stay tuned for more updates and explorations into the fascinating realm of privacy-aware data science.


---

## Annexes

### Glossary of Terms

- **Differential Privacy**: A mathematical framework for quantifying the privacy guarantees provided by a data analysis or processing mechanism.
- **Privacy Metric**: A measure used to assess the level of privacy protection afforded to individuals within a dataset.
- **Data Anonymization**: The process of removing or obfuscating identifying information from a dataset to protect the privacy of individuals.

### References

1. Warner SL. Randomized response: a survey technique for eliminating evasive answer bias. J Am Stat Assoc. 1965 Mar;60(309):63-6. PMID: 12261830.
2. John C. Duchi, Michael I. Jordan, and Martin Wainwright. Local Privacy and Minimax Bounds: Sharp Rates for Probability Estimation. Advances in Neural Information Processing Systems (2013)
111. Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3-4), 211-407.
222. Narayanan, A., & Shmatikov, V. (2008). Robust de-anonymization of large sparse datasets. In Security and Privacy, 2008. SP 2008. IEEE Symposium on (pp. 111-125). IEEE.


---

<style>
.highlight {
  background-color: red;
}
.highlight-on-hover:hover {
        background-color: yellow;
    }
/* Quiz form styles */
.quiz-form {
        max-width: 500px;
        margin: auto;
        padding: 20px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f9f9f9;
}

.quiz-question {
        margin-bottom: 20px;
}

.quiz-options label {
        display: block;
        margin-bottom: 10px;
}

.quiz-submit {
        background-color: #4caf50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
}

.quiz-submit:hover {
        background-color: #45a049;
}

/* Quiz results styles */
.quiz-results {
        margin-top: 20px;
        font-weight: bold;
}
.quiz-options label {
        display: block;
        margin-bottom: 10px;
    }
.quiz-options label.correct {
        color: green;
}
.quiz-options label.incorrect {
        color: red;
}
a[name]:hover {
        background-color: yellow; /* Change to the same color as normal state to maintain yellow highlight */
        text-decoration: none; /* Optionally remove underline on hover */
}
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

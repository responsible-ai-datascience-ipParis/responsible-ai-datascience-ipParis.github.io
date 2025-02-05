+++
title = 'How to augment my dataset?'
date = 2025-02-05T14:03:13+01:00
draft = false
+++
<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$','$$'], ['\\[','\\]']],
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

# Authors : *Tristan Waddington, Fabien Lagnieu & Dimitri Henrard-Iratchet*

### Paper: Comment on the research paper: [**Tailoring Mixup to Data for Calibration**](https://arxiv.org/abs/2311.01434), written by *Quentin Bouniot, Pavlo Mozharovskyi & Florence d’Alché-Buc*, from LTCI, Télécom Paris, Institut Polytechnique de Paris, France



TODO: insert table of contents

# Introduction

<p align="center">
  <figure>
  <img src="/images/MixUpDataCalibration/AI_introvert.png" 
    alt="Test image mathematical"
    width=300>
    </img>
  <figcaption>The power of AI.</figcaption>
  </figure>
</p>
# Main part

Test LaTeX : $\sigma \left( \frac{1}{2}\right)$


$$\mathbb{R}^3 $$


## Code Test
```python
import numpy as np
ones = np.ones(1)
print(ones)
```
```bash
output:
>>> [[1]]
```
# Conclusion

**To explain**:
- Similarity kernel
- Beta distributions
- different procedures of data augmentation 
- produce images of augmentation
- compare with other methodes : results and computing power
- Why "calibration"?

Further idea: Compare with CutMix[^Yun] and Manifold Mixup[^Verma]

# References

[^Verma]: Verma, V., Lamb, A., Beckham, C., Najafi, A., Mitliagkas, I., Lopez-Paz, D., and Bengio, Y. (2019).
Manifold mixup: Better representations by interpolating hidden states. In Chaudhuri, K. and
Salakhutdinov, R., editors, Proceedings of the 36th International Conference on Machine Learning,
volume 97 of Proceedings of Machine Learning Research, pages 6438–6447. PMLR.

[^Yun]: Yun, S., Han, D., Chun, S., Oh, S. J., Yoo, Y., and Choe, J. (2019). Cutmix: Regularization strategy
to train strong classifiers with localizable features. In 2019 IEEE/CVF International Conference
on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages
6022–6031. IEEE.

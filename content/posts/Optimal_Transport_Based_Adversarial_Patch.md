+++
title = 'Optimal_Transport_Based_Adversarial_Patch'
date = 2024-02-03T22:22:36+01:00
draft = false
+++


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

        inlineMath: [['$','$'], ['\','\']],
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

# OPTIMAL TRANSPORT BASED ADVERSARIAL PATCH ATTACKS
## Authors:
* Mohammed Jawhar
* Aymane Rahmoune

## Introduction

Imagine you're showing a photo to a friend, asking them to guess who's in it. Now, imagine sticking a tiny, almost invisible sticker on that photo. For some reason, this sticker makes your friend completely unable to recognize who's in the picture. This might sound like a magic, but something similar can happen with Computer Vision models designed to capture an image content, either through a classification, a segmentation or even a generation task. These AI programs can be vulnerable to such tricks, which we call in the scientific domain, Adversarial Patch Attacks.

As AI becomes increasingly integrated into various aspects of our lives, including critical applications like passport security systems, autonomous vehicles, traffic sign detection, and surgical assistance, the reliability, trustworthiness, and performance of these systems under all conditions became of prime importance. This has led to a growing interest in the area of Robust AI, which focuses on enhancing the safety and security of AI technologies by improving their resilience to adverse conditions and digital threats. Within this domain, the study of Attacks and Defense ways plays a pivotal role.


![ScRoad_scene](http:/localhost:1313/images//image_optimal_transport_patch/road_scene.png)

While these attacks might not seem like a big deal, nor dangerous in this context, the consequences can be severe in critical scenarios - take for example an autonomous vehicle failing to recognize a stop sign, hurting potentially a pedestrian. In this blog we will explore a new approach used for developping such adversarial patch attacks, based on Optimal Transport, as outlined in the paper ....... We will try to follow the same structure as in the paper to make the reading easier for you, but with much more simplicity.


## Understanding Adversarial Attacks

First thing first, let us redefine some previously mentionned concepts with their "technical" definition, while making them into context.

As deep neural networks keep getting better, developers are working hard to make sure they are trustworthy and reliable. This means constantly testing them to see how well they can handle different challenges, quantifying their robustness, and developping some robustification methods. In the context of image classification for instance, one way to do this is by designing adversarial attacks, which consists of a perturbation or noise, sometimes invisible patterns added to the input images in order to confuse the model and make it misclassify them, causing a huge drop in the accuracy.

Adversarial Patch Attacks - a specific type of these attacks - consists of altering only a small part(patch) of the input, either physically or digitally by inserting a crafted "sticker". This type of attacks is what we call Adversarial Patch Attacks, and it happens to be more threatful as it can be easily applied in real life as they do not require modification of the entire image. Some go further by trying to test these engineered adversarial patches on various target models, beyond the original one used for learning, to evaluate the attack transferability and ameliorate its efficacy making it more challenging to surpass. 

Despite the fact that crafting adversarial patch attacks is mainly based around maximizing the classification error through a gradient ascent, we can differenciate between three distinct approaches:

![APA_strategies](http:/localhost:1313/images/image_optimal_transport_patch/APA_strategies.png)

* **Decision boundaries based :** Which is the most applied approach in previous works and litterature. It focuses on pushing the image's representation in the neural network's **decision** space, across the decision boundary, making the network perceive it as belonging to a different, probability maximized class.

  * To simplify this approach, imagine a group of fans attempting to sneak into a VIP section at a concert by dressing in a fancy way, like known VIP guests(targeted class). The idea is to blend in so well that they are indistinguishable from actual VIPs to the security guards (the ML model). Despite the simplicity and goodness of this strategy, it has some drawbacks :

    * It is highly dependant on the model on which the attack is based, which makes it not really transferable: The success of this method hinges on the security's lack of detail. If they are controlled by another security gard who is very familiar with the actual VIPs, the disguises will fail. 

    * The patch may push the corrupted image representations into unknown regions of the representation space: In their attempt to mimic the VIPs, there's a risk that their disguises might be so overdone that they don't resemble any actual VIPs, pushing them to have a weird unique look. Hence, they end up in a no-man's-land, not fitting in with either the regular attendees or the VIPs.


* **Feature point based :** Instead of crossing a decision boundary, this strategy aims to modify the input so its representation in the **feature space** matches the one of a target point belonging to a different class. This is like fine-tuning the attack to match a specific "signature" that the model associates to a specific point.

  * Revisiting our concert analogy, consider the fans now opting to mimic a specific celebrity known to be attending the concert, assuming that matching this one high-profile individual's appearance will guarantee them entry. Although it seems more precise and effective than the first approach, this strategy has a significan drawback :

    * It depends heavily on the targeted point selection, this later may be not representative of all instances in the target class :  For instance, if the celebrity is known for a distinctive but uncommon style or if it's unusual for such celebrities to attend such events, their attempt to copy him might not match what the security team usually expects from VIP guests.

* **Distributed based :** This new approach implemented in the paper we are analyzing , is based on Optimal Transport theory, and aims to alter the overall feature distribution ofa set of input images belonging to a specific class, to resemble another class's distribution, reducing the gap between them in the **feature space**. It is more sophisticated than the previous ones as it exploits the fundamental way neural networks process and classify images based on learned distributions.

  *  This time, the group studies a wide variety of guests behaviors and appearances to craft a new, ambiguous look that doesn't specifically mimic any single guest type, nor disguise blindly in a "VIP" style, but instead blends into the overall crowd, avoiding easy detection.

     * The main advantage of this approach is that it allows a better transferability between models, enhancing the performance in the blackbox configuration, as it is independant of the classifier's decision boundary , and the choice of a specific target point. Furthermore it captures the useful characteristics (features) from an input in a more universal way.





  
   







































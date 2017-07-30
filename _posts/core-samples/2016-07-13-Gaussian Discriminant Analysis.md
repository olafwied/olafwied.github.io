---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Gaussian distribution, discriminant analysis, nearest shrunken centroid]
mathjax: true
---
{% include JB/setup %}

## Gaussian Discriminant Analysis

In the [last post]({{ site.baseurl }}{% post_url /core-samples/2016-06-30-Bayesian Basics and Naive Bayes %}) we talked about class conditional densities in generative classifiers and Naive Bayes. This post will address Gaussian discriminant analysis and how it relates to these concepts.

### Overview

Gaussian discriminant analysis is a technique where we assume the class conditional densities to be Gaussian:

$$P(x \mid y=c, \theta) = N(x \mid \mu_c,\sigma_c)$$

Naive Bayes assumes that the features $x$ are conditionally independent on the class label $c$. Thus, we get a Gaussian flavored Naive Bayes approach if $\sigma_c$ is diagonal!

From our previously explored equation for generative classifiers, we can easily derive a decision rule for a new feature vector:

$$\widehat{y}(x) = argmax_c \, (log P(y=c \mid \pi) + log P(x \mid \theta_c))$$

### Quadratic Decision Surface

To understand why this model is called Quadratic Discriminant Analysis, let's simply plug in the definition of the multivariate Gaussian density using the Bayes' rule (again see the [last post]({{ site.baseurl }}{% post_url /core-samples/2016-06-30-Bayesian Basics and Naive Bayes %})) and let $\pi_c$ be the class probability $P(y=c)$ (estimated as the proportions of instances in class $c$):

$$ P(y=c \mid x,\theta) = \frac{\pi_c \mid 2 \pi \mu_c \mid^{-\frac{1}{2}} exp\left( -\frac{1}{2}(x - \mu_c)^T \mu_c^{-1} (x-\mu_c)\right)}{\sum_{c_k} \pi_{c_k} \mid 2 \pi \mu_{c_k} \mid^{-\frac{1}{2}} exp\left( -\frac{1}{2}(x-\mu_{c_k})^T \mu_{c_k}^{-1}(x-\mu_{c_k})\right)} $$

This gives a function that is quadratic in $x$!


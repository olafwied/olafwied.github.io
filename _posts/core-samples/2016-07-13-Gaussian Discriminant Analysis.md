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

$$P(x \mid y=c, \theta) = N(x \mid \mu_c,\sigma_c)$$ with mean (vector) $\mu_c$ and covariance (matrix) $\sigma_c$.

Naive Bayes assumes that the features $x$ are conditionally independent on the class label $c$. Thus, we get a Gaussian flavored Naive Bayes approach if $\sigma_c$ is diagonal!

From our previously explored equation for generative classifiers, we can easily derive a decision rule for a new feature vector:

$$\widehat{y}(x) = argmax_c \, (log P(y=c \mid \pi) + log P(x \mid \theta_c))$$

### Quadratic Decision Surface

To understand why this model is called Quadratic Discriminant Analysis, let's simply plug in the definition of the multivariate Gaussian density using the Bayes' rule (again see the [last post]({{ site.baseurl }}{% post_url /core-samples/2016-06-30-Bayesian Basics and Naive Bayes %})) and let $\pi_c$ be the class probability $P(y=c)$ (estimated as the proportions of instances in class $c$):

$$ P(y=c \mid x,\theta) = \frac{\pi_c \mid 2 \pi \sigma_c \mid^{-\frac{1}{2}} exp\left( -\frac{1}{2}(x - \mu_c)^T \sigma_c^{-1} (x-\mu_c)\right)}{\sum_{c_k} \pi_{c_k} \mid 2 \pi \sigma_{c_k} \mid^{-\frac{1}{2}} exp\left( -\frac{1}{2}(x-\mu_{c_k})^T \sigma_{c_k}^{-1}(x-\mu_{c_k})\right)} $$

Thresholding this formula gives a function that is quadratic in $x$! (e)

### From Quadratic to Linear

One convenient assumption that will lead to Linear Discriminant Analysis is where all classes share the same covariance matrix ($\sigma_c = \sigma \forall c$). The first simplification tot notice is that the term $\mid 2 \pi \sigma_c \mid$ will cancel out with the denominator. Further, we are interested in thresholding the probabilty by comparing $P(y=c_1 \mid x,\theta) > P(y=c_2 \mid x,\theta)$. Since they share the same denominator, we will focus on the numerators:

$$P(y=c \mid x,\theta) \sim \pi_c exp\left( \mu_c^T \sigma^{-1}x -\frac{1}{2}x^T\sigma^{-1}\x -\frac{1}{2}\mu_c^T\sigma^{-1}\mu_c\right)$$

We can now isolate the quadratic term (which will then cancel out when we do thresholding):

$$= exp\left( \mu_c^T\sigma^{-1}x - \frac{1}{2}\mu_c^T\sigma^{-1}\mu_c + log\pi)c\right) exp\left(-\frac{1}{2}x^T\sigma^{-1}x\right)


---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Gaussian distribution, discriminant analysis, nearest shrunken centroid]
mathjax: true
---
{% include JB/setup %}

## Gaussian Discriminant Analysis

### Overview
In the [last post]({{ site.baseurl }}{% post_url /core-samples/2016-06-30-Bayesian Basics and Naive Bayes %}) we talked about class conditional densities in generative classifiers and Naive Bayes. This post will address Gaussian discriminant analysis and how it relates to these concepts.

Gaussian discriminant analysis is a technique where we assume the class conditional densities to be Gaussian:

$$P(x \mid y=c, \theta) = N(x \mid \mu_c,\sigma_c)$$

Naive Bayes assumes that the features $x$ are conditionally independent on the class label $c$. Thus, we get a Gaussian falvored Naive Bayes approach if $\sigma_c$ is diagonal!

From our previously explored equation for generative classifiers, we can easily derive a decision rule for a new feature vector:

$$\hat{y}(x) = argmax_c (log P(y=c \mid \pi) + log P(x \mid \theta_c))$




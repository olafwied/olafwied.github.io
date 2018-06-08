---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Gaussian, Discriminant Analysis]
mathjax: true
---
{% include JB/setup %}

## Gaussian Discriminant Analysis

In the [last post]({{ site.baseurl }}{% post_url /core-samples/2016-06-30-bayesian basics and naive bayes%}) we talked about class conditional densities in generative classifiers and Naive Bayes. This post will address Gaussian discriminant analysis and how it relates to these concepts.

### Overview

Gaussian discriminant analysis is a technique where we assume the class conditional densities to be Gaussian:

$$P(x \mid y=c, \theta) = N(x \mid \mu_c,\sigma_c)$$ with mean (vector) $\mu_c$ and covariance (matrix) $\sigma_c$.

Naive Bayes assumes that the features $x$ are conditionally independent on the class label $c$. Thus, we get a Gaussian flavored Naive Bayes approach if $\sigma_c$ is diagonal!

From our previously explored equation for generative classifiers, we can easily derive a decision rule for a new feature vector:

$$\widehat{y}(x) = argmax_c \, (log P(y=c \mid \pi) + log P(x \mid \theta_c))$$

### Quadratic Decision Surface

To understand why this model is called Quadratic Discriminant Analysis (QDA), let's simply plug in the definition of the multivariate Gaussian density using the Bayes' rule (again see the [last post]({{ site.baseurl }}{% post_url /core-samples/2016-06-30-bayesian basics and naive bayes%})) and let $\pi_c$ be the class probability $P(y=c)$ (estimated as the proportions of instances in class $c$):

$$ P(y=c \mid x,\theta) = \frac{\pi_c \mid 2 \pi \sigma_c \mid^{-\frac{1}{2}} exp\left( -\frac{1}{2}(x - \mu_c)^T \sigma_c^{-1} (x-\mu_c)\right)}{\sum_{c_k} \pi_{c_k} \mid 2 \pi \sigma_{c_k} \mid^{-\frac{1}{2}} exp\left( -\frac{1}{2}(x-\mu_{c_k})^T \sigma_{c_k}^{-1}(x-\mu_{c_k})\right)} $$

Thresholding this formula gives a function that is quadratic in $x$!

### From Quadratic to Linear

One convenient assumption that will lead to Linear Discriminant Analysis (LDA) is where all classes share the same covariance matrix ($\sigma_c = \sigma \forall c$). The first simplification tot notice is that the term $\mid 2 \pi \sigma_c \mid$ will cancel out with the denominator. Further, we are interested in thresholding the expression by comparing e.g. $P(y=c_1 \mid x,\theta) > P(y=c_2 \mid x,\theta)$. Since they share the same denominator, they will cancel out and we focus on what is remaining of the numerators:

$$P(y=c \mid x,\theta) \propto \pi_c exp\left( \mu_c^T \sigma^{-1}x -\frac{1}{2}x^T\sigma^{-1}x -\frac{1}{2}\mu_c^T\sigma^{-1}\mu_c\right)$$

We can now isolate the quadratic term (which will then cancel out when we do thresholding):

$$\ldots = exp\left( \mu_c^T\sigma^{-1}x - \frac{1}{2}\mu_c^T\sigma^{-1}\mu_c + log\pi_c\right) exp\left(-\frac{1}{2}x^T\sigma^{-1}x\right)$$

Taking the logarithm now gives a linear expression. The decision boundary between two classes is therfore a straight line. In the binary case, this would mean projecting $x$ on the line through the class means, say $\mu_{c_1}$ and $\mu_{c_2}$ and check to which of the means the projection is closer.

### Remarks on Model Fitting
Fitting the model can be done using the Maximum Likelihood Estimates (MLE), that are (as so often) the empirical class sample means and covariances. As mentioned, the priors can be estimated as the proportion of instances in each class.

There are several methods to prevent the MLE from overfitting. The basic idea is to simplify the covariance matrix that can sometimes be ill-conditioned:

- Diagonal covariance matrix (see above, Naive Bayes)
- Shared covariance matrices (leading to LDA)
- Shared and diagonal ("diagonal LDA")
- Fit a full covariance matrix (using [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) estimates or marginalization) using an appriopriate prior like Laplace (that promotes sparsity) or inverse Wishart (that leads to regularization)
- ...other simplifying techniques like dimensionality reduction.

See for example Murphy's book on Machine Learning for more details on these techniques.


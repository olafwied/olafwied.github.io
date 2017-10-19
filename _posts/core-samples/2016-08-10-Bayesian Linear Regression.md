---
layout: post
category : Machine Learning
tagline: hands-on
tags : [Linear Regression, Regularization, Regularization, Gaussian, Prior]
mathjax: true
---
{% include JB/setup %}

## Bayesian Linear Regression 

In the last post, we have discussed linear regression and how a Gaussian prior can reduce overfitting by encouraging smaller weights. We have also seen how to produce effective point estimates.

In this post we will explore to some extent how to compute the full distributions over the weights. This allows to explicitely model uncertain about our model.

Since the computation is quite involved, we will only consider the case where the variance is known. However, the results with unnkonw variance are very similar. 

### Linear Gaussian Systems

To compute the posterior, we need a result from the theory of multivariate Gaussians, in particular about what is called Linear Gaussian Systems.

A linear Gaussian system is defined through two variables $x$ and $y$ with the  following distributions:

$$P(x) = N(x \mid \mu_x, \Sigma_x)$$ and

$$P(y\mid x) = N(y \mid A x + b, \Sigma_y)$$. 

We can think of $y$ as a noisy version of $x$. The result we need tells us how to infer $x$ from $y$:

$$ P(x\mid y) = N(x \mid \mu_{x\mid y}, \Sigma_{x\mid y})$$ 

with

$$ \Sigma_{x\mid y}^{-1} = \Sigma_x^{-1} + A^T \Sigma_y^{-1}A$$

and

$$\mu_{x\mid y} = \Sigma_{x\mid y} \left( A^T \Sigma_y^{-1} (y-b)+\Sigma_x^{-1}\mu_x \right)$$

For a proof see e.g. section 4.4.3 of K. Murphy's Machine Learning book.

### The Posterior for Bayesian Linear Regression
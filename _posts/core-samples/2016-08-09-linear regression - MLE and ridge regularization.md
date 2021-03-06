---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Linear Regression, Ridge, Regularization, MLE, Likelihood, Subspace, Regularization, Gaussian, Prior]
mathjax: true
---
{% include JB/setup %}

## Linear Regression 

Linear regression is one of the fundamental and commonly used models of supervised machine learning and statistics. Linear regression models dependent variables via a linear combination of explanatory variables.

### Model

The model is of the following general form:

$$P(y \mid x,\theta) = N(y \mid w^T\phi(x),\sigma^2) $$

where $\phi(x)$ is a potentially non-linear function of the inputs $x$. The models remains linear in the weights $w$!

### Estimation

The most popular way to fit a linear regression model is probably via MLE. Maximum Likelihood estimation leads to the least squares method. To understand the term least squares let's examine the log-likelihood of the linear regression model:

If we assume that the training examples are independent and identically distributed (iid) we get

$$L(\theta) = \sum_{i=1}^N \log P(y_i \mid x_i,\theta) =\sum_{i=1}^N \log \left[ \left( \frac{1}{2\pi\sigma^2}\right)^{\frac{1}{2}} \exp \left( -\frac{1}{2\sigma^2}(y_i - w^Tx_i)^2\right)\right] \\
=-\frac{1}{2\sigma^2}RSS(w) - \frac{N}{2}\log(2\pi\sigma^2)$$.

Hence, we find that MLE gives us the solution that minimizes the Residual Sum of Squares $RSS(w) = \sum_{i=1}^N (y_i - w^Tx_i)^2$, which equals the $L_2$-norm of the residuals!

Using matrix algebra it can be quickly verified that the solution is given by

$$\hat{w} = (X^T X)^{-1}X^Ty $$

Predictions $\hat{y}$ are given by $X\hat{w} = X(X^TX)^{-1}X^Ty$. If you are familiar with linear algebra you might be able to recognizie that this corresponds to an orthogonal projection of $y$ onto the column space of $X$. (This is not too surprising. The features (columns) span a linear subspace embedded in $N$-dimensional space given that we have more training samples than features. Linear regression now tries to find the closest vector to $y$ in that subspace which is of course given by it's orthogonal projection!)

Another thing to note is that the $RSS$ or more general the (negative) log likelihood is convex (a parabola). This ensures that we always find a global optimum. (This can also be seen from our discussion about orthogonal projections.)

### Ridge Regression

The biggest problem of MLE is that it can easily lead to overfitting. There are many ways to tackle this problem. Here, I will only address one popular and very effective idea: Ridge regression or $L_2$-regularization.

Ridge regression encourages small weights by putting a Gaussian prior with mean zero on the weights: $P(w) = \prod_j N(w_j\mid 0,\alpha^2)$. 

Similar to above, we can show that MLE is equivalent to minimizing 

$$\frac{1}{N}\sum_{i=1}^N(y_i - w^Tx_i)^2 + \lambda \sum_{i=1}^N w_i^2$$

with $\lambda = \frac{\sigma^2}{\alpha^2} \geq 0$. The second term ecnourages small weights. This is also known as weight decay. 

If your data is not normalized and you need an offset parameter $w_0$, it will be outside the penality term since it doesn't affect the complexity of the function!

#### Beyond Point Estimates

In the [next post]({{ site.baseurl }}{% post_url /core-samples/2016-08-10-bayesian linear regression%}) we will use Bayesian analysis to compute the full posterior over $w$ and $\sigma^2$.



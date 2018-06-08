---
layout: post
category : Machine Learning
tagline: hands-on
tags : [Gaussian Mixture Model, Bayes, Latent Variables, Expectation-Maximization, Kullback-Leibler]
mathjax: true
---
{% include JB/setup %}

The previous post gave a gentle introduction to the EM algorithm and the intuition behind it. 
In this post, I will lay out some more mathematical details on how to perform the E and M step for general latent-variable models.

## The General Form of Expectation-Maximization

We want to maximize the (log) likelihood of our data with respect to the model parameters $P(x \mid \theta)$ where we assume our data $X$ to be $N$ i.i.d. samples and therefore

$$ log P(X \mid \theta) = \sum_{i=1}^N log P(x_i \mid \theta)$$. 

We can introduce our latent variable here using the law of total proabbility through a sum or an integral:

$$\sum_{i=1}^N log P(x_i \mid \theta) = \sum_{i=1}^N \sum_{c=1}^C log P(x_i, t_i=c \mid \theta)$$. You can think of $T$ as our latent variable from the Gaussian Mixture Model from the previous post.

This function could be (locally) optimized with a gradient decent routine. However, with EM we can usually do better and find a solution faster.

The idea of EM is to find an optimal (in the sense explained below) lower bound on the expression above that can be easily optimized. This is done using Jensen's inequality:

$$\sum_{i=1}^N \sum_{c=1}^C log P(x_i, t_i=c \mid \theta) = \sum_{i=1}^N \sum_{c=1}^C q(t_i=c)\frac{log P(x_i, t_i=c \mid \theta)}{q(t_i=c)} $$ where $q$ is any distribution over $T$.
Applying Jensen's inequalty, we get

$$log P(X \mid \theta) \geq \sum_{i=1}^N \sum_{c=1}^C q(t_i=c) log \frac{P(x_i, t_i=c \mid \theta)}{q(t_i=c)} = \mathcal(L)(\theta, q)$$.

We derived a family of lower bounds for the log-likelihood that depends on $q$ and $\theta$. 





## The Variational Lower Bound

## The E-Step

## The M-Step

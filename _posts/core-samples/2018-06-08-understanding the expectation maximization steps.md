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

## The Variational Lower Bound

We want to maximize the (log) likelihood of our data with respect to the model parameters $P(x \mid \theta)$ where we assume our data $X$ to be $N$ i.i.d. samples and the latentn variable $T$ can take $C$ values. Hence,

$$ log P(X \mid \theta) = \sum_{i=1}^N log P(x_i \mid \theta)$$. 

We can introduce our latent variable here using the law of total proabbility through a sum or an integral:

$$\sum_{i=1}^N log P(x_i \mid \theta) = \sum_{i=1}^N \sum_{c=1}^C log P(x_i, t_i=c \mid \theta)$$. 

You can think of $T$ as our latent variable from the Gaussian Mixture Model from the previous post.

This function could be (locally) optimized with a gradient decent routine. However, with EM we can usually do better and find a solution faster.

The idea of EM is to find an optimal (in the sense explained below) lower bound on the expression above that can be easily optimized. This is done using Jensen's inequality:

$$\sum_{i=1}^N \sum_{c=1}^C log P(x_i, t_i=c \mid \theta) = \sum_{i=1}^N \sum_{c=1}^C q(t_i=c)\frac{log P(x_i, t_i=c \mid \theta)}{q(t_i=c)} $$ where $q$ is any distribution over $T$.
Applying Jensen's inequalty, we finally get

$$log P(X \mid \theta) \geq \sum_{i=1}^N \sum_{c=1}^C q(t_i=c) log \frac{P(x_i, t_i=c \mid \theta)}{q(t_i=c)} = \mathcal{L}(\theta, q)$$.

In fact, we derived a family of lower bounds for the log-likelihood that depends on $q$ and $\theta$. 

## The General Form of Expectation-Maximization

We now use the ideas developed in the previous post of optimizing it by alternating between finding the best $q$ and finding the best $\theta$: 

While $\mathcal{L}(\theta^j, q^j) > tol \cdot \mathcal{L}(\theta^{j-1}, q^{j-1})$:

#### E-Step

Find $q^{j+1}$ that maximizes $\mathcal{L}(\theta^j, q) = \mathcal{L}_{\theta^j}(q)$. 

#### M-Step

Find $\theta^{j+1}$ that maximizes $\mathcal{L}(\theta, q^{j+1}) = \mathcal{L}_{q^{j+1}}(\theta)$.

Let's now discuss the details of each step.

## The E-Step

Maximizing $\mathcal{L}(\theta^j, q)$ w.r.t. $q$ is the same (by definition of the lower bound) as minimizing the difference between $\mathcal{L}(\theta^j, q)$ and the log-likelihood $log P(X \mid \theta)$. Plugging in the definition of the variational lower bound we get the following:

$$log P(X \mid \theta) - \mathcal{L}(\theta, q) = \sum_{i=1}^N log P(x_i \mid \theta) - \sum_{i=1}^N \sum_{c=1}^C q(t_i=c) log \frac{P(x_i, t_i=c \mid \theta)}{q(t_i=c)}$$ 

Using the fact that $\sum_{c=1}^C q(t_i=c) = 1$, we can continue as follows:

$$= \sum_{i=1}^N log P(x_i \mid \theta) \sum_{c=1}^C q(t_i=c) - \sum_{i=1}^N \sum_{c=1}^C q(t_i=c) log \frac{P(x_i, t_i=c \mid \theta)}{q(t_i=c)}$$

(changing the summation order)

$$= \sum_{i=1}^N  \sum_{c=1}^C log P(x_i \mid \theta) q(t_i=c) - \sum_{i=1}^N \sum_{c=1}^C q(t_i=c) log \frac{P(x_i, t_i=c \mid \theta)}{q(t_i=c)}$$

(exploiting the rules of the logarithm)

$$= \sum_{i=1}^N  \sum_{c=1}^C q(t_i=c) log \frac{P(x_i \mid \theta) \cdot q(t_i=c)}{P(x_i, t_i=c \mid \theta)}$$

(and with $P(x_i, t_i=c \mid \theta) = P(t_i=c \mid x_i, \theta) \cdot P(x_i \mid \theta)$)

$$= \sum_{i=1}^N  \sum_{c=1}^C q(t_i=c) log \frac{q(t_i=c)}{P(t_i=c \mid x_i, \theta)}$$

(see [this post]({{ site.baseurl }}{% post_url /core-samples/2016-07-02-kullback-leibler and the gaussian distribution%}) for a reminder of the Kullback-Leibler divergence)

$$= \sum_{i=1}^N  D_{KL}(q(t_i) \mid \mid P(t_i \mid x_i, \theta)$$

We established that the distance between the log-likelihood and the lower bound can be expressed in terms of the Kullback-Leibler divergence $D_{kL}$. We know that $D_{KL} \geq 0$ and $D_{KL}=0$ when $q(t_i) = P(t_i \mid x_i, \theta)$ and so we have found our optimum for $q$.

In summary, the posterior probability of $t_i$ given the data and model parameters gives us the optimium for the E-step!

## The M-Step

Let's rewrite $\mathcal{L}$ to maximizie it w.r.t. to $\theta$:

$$\mathcal{L}(\theta, q) = \sum_{i=1}^N  \sum_{c=1}^C q(t_i=c) log \frac{P(x_i, t_i=c \mid \theta)}{q(t_i=c)}$$

(expanding the fraction in the $log$)

$$= \sum_{i=1}^N  \sum_{c=1}^C q(t_i=c) log P(x_i, t_i=c \mid \theta)q(t_i=c) - q(t_i=c) log q(t_i=c)$$

(since the last term does not depend on $\theta$, we can ignore during maximization)

$$= E_q log P(X,T \mid \theta) + const$$.

Often times, e.g. for a Gaussian distribution or if otherwise properly chosen, this function is relatively easy to optimize (or even concave with a global optimum). 

## The General Form of Expectation-Maximization (revisisted)

While $\mathcal{L}(\theta^j, q^j) > tol \cdot \mathcal{L}(\theta^{j-1}, q^{j-1})$:

#### E-Step

$q^{j+1} = P(t_i \mid x_i, \theta)^j$

#### M-Step

$\theta^{j+1} = argmax_{\theta} E_{q^{j+1}} log P(X,T \mid \theta)$.

## Convergence Properties of the EM Algorithm

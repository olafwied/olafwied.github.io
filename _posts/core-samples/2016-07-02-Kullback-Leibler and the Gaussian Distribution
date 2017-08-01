---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Gaussian, Kullback-Leibler, Information Gain, Entropy, Mutual Information]
mathjax: true
---
{% include JB/setup %}

In this article, we explore another reason (besides the central limit theorem and historical reasons) why the Gaussian Distribution is so widely used and often our first choice.
Given only the first two moments, or the expected value and the variance, the Gaussian Distribution has maximum entropy. That means it is the distribution that makes the least additional asssumptions.

## Entropy and Kullback-Leibler Divergence

Let's start by defining some terms. The entropy $H$ of a random variable $X$ is defined as 

$$H(X) = -\sum_i^n P(x_i)\log_b P(x_i)$$, where $0 \log_b(0)$ is set to $0$. (For continous variables, replace sums with integrals.)
The unit of $H$ depends on the logarithmic base $b$ and is called bit (or shannon) for $b=2$ and nat for the natural logarithm.

This expression encodes the expected amount of information (measured in bits, nats etc.) from each _event_ of $X$. This is easiest explained with the standard example of a coin flip:
If the coin is fair, i.e. a 50/50 chance of head or tail, each coin flip generates $-log_b(0.5) = 1$ bit of information. However, if the coin is biased, say 20/80, we obtain $2.3$ bits for heads (20% chance) and only $0.3$ bits for tails (80%).
The more likely an event, the less suprised we are. We now average the information content of all events and weigh them with their probability. 
It is straight forward to see, that the entropy is maximized for a fair coin. The reason is, that if heads and tails have equal probability, it is the most difficult to predict the next toss. We obtain one bit of information for every toss.
However, if the coin is biased, we receive less than one bit of information per toss. For our 20/80 coin, we get 

$$H \approx 0.2 \cdot 2.3 + 0.8 \cdot 0.3 = 0.7$$

We know that tails is more likely than heads, reducing our uncertainty for the next toss compared to a fair coin.

The Kllback-Leibler divergence $D_{KL}$, on the other hand, is a measure of difference between two probability distributions. It is defined as

$$D_{KL}(P\mid\mid Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}$$,

which is the expected, logarithmic difference between the probability distributions $P$ and $Q$.

We can re-express is in terms of entropy as follows

$$D_{KL}(P\mid\mid Q) = - \sum_x P(x)\log Q(x) + \sum_x P(x)\log P(x) = H(P,Q) - H(P)$,

where $H(P,Q)$ is called the cross-entropy. Following the logic of our coin experiment, this means measuring the expected information content of $X$ if the information in now encoded under some alternative coding "scheme" $Q$ instead of the true coding scheme $Q$.

## Maxmium Entropy Property of the Gaussian Distribution





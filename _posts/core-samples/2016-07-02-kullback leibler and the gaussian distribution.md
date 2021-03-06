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

### Entropy

Let's start by defining some terms. The entropy $H$ of a random variable $X$ is defined as 

$$H(X) = -\sum_x P(x)\log_b P(x),$$

where $0 \log_b(0)$ is set to $0$. (For continous variables, replace sums with integrals.)
The unit of $H$ depends on the logarithmic base $b$ and is called bit (or shannon) for $b=2$ and nat for the natural logarithm.

This expression encodes the expected amount of information (measured in bits, nats etc.) from each _event_ of $X$. This is easiest explained with the standard example of a coin flip:

If the coin is fair, i.e. a 50/50 chance of head or tail, each coin flip generates $-log_b(0.5) = 1$ bit of information. However, if the coin is biased, say 20/80, we obtain $2.3$ bits for heads (20% chance) and only $0.3$ bits for tails (80%).
The more likely an event, the less suprised we are. 

We now average the information content of all events and weigh them with their probability. 
It is straight forward to see, that the entropy is maximized for a fair coin. The reason is, that if heads and tails have equal probability, it is the most difficult to predict the next toss. We obtain one bit of information for every toss.

However, if the coin is biased, we receive less than one bit of information per toss _on average_. For our 20/80 coin, we get 

$$H \approx 0.2 \cdot 2.3 + 0.8 \cdot 0.3 = 0.7.$$

We know that tails is more likely than heads, reducing our uncertainty for the next toss compared to a fair coin.

### Kullback-Leibler Divergence

The Kllback-Leibler divergence $D_{KL}$, on the other hand, is a measure of difference between two probability distributions. It is defined as

$$D_{KL}(P\mid\mid Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)},$$

which is the expected, logarithmic difference between the probability distributions $P$ and $Q$.

We can re-express is in terms of entropy as follows

$$D_{KL}(P\mid\mid Q) = - \sum_x P(x)\log Q(x) + \sum_x P(x)\log P(x) = H(P,Q) - H(P),$$

where $H(P,Q)$ is called the cross-entropy. Following the logic of our coin experiment, this means measuring the expected information content of $X$ if the information in now encoded under some alternative coding "scheme" $Q$ instead of the true coding scheme $P$.

### Maxmium Entropy Property of the Gaussian Distribution

Equipped with these basic terms, we can now specify _and prove_ the maximum entropy property of the Gaussian distribution.


> The Gaussian distribution has maxmium entropy under the constraints of equal covariance matrices:
> Let $g \sim N(0,\Sigma)$ and $f$ s.t. $\int f(x)x_i x_j dx = \Sigma_{ij}$. Then, $H(g) \leq H(f)$.

$$ 0 \leq D_{KL}(g\mid \mid f) = \int g(x) \log\left(\frac{g(x)}{f(x)}\right)dx$$

$$ = -H(g) -\int g(x)\log \left( f(x) \right)dx = -H(g) - \int f(x) \log\left(\frac{g(x)}{f(x)}\right)dx$$

$$=- H(g) + H(f) $$

which yields the result. We can switch $f$ for $g$ since they both yield the same quadratic form given through $\log \left( f(x) \right)$.

Similarly, one can show that the expenontial distribution has maximum entropy over the positive values under the constraints of means!

Another result, that is very clear intuitively from our earlier thoughts on the coin flip experiment, is the following: On a fixed interval $\[a,b\]$ without constraints, the uniform distribution over the given interval has maximum entropy. No area over the interval has higher propability than others leading to maximum 'uncertainty'.

#### References
[Why the Normal Distribution? Paul Rojas (2010)](http://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/Gaussian-distribution.pdf): A short, easy to follow summary on the significance of the Gaussian distribution.

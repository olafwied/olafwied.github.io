---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Bayes, prior, posterior, likelihood, document classification]
mathjax: true
---
{% include JB/setup %}

## Bayesian Basics 

### Generative Models using Bayes' Rule
The Bayes' rule 

$$
\begin{eqnarray*}P(X=x \mid Y=y) = \frac{ P(X=x, Y=y) }{ P(Y=y)} &=&  
\frac{P(X=x)P(Y=y \mid X=x)}{\sum_{z_x}P(X=z_x)P(Y=y \mid X=z_x)} \end{eqnarray*}
$$

is one of the fundamental equations in probability theory and also in machine learning. One reason is that we can use it to build **generative** classifier as follows: 

Given a feature vector $$x$$, a class label $$c$$ and model described by parameters $$ \theta $$, we can compute the probabilty to observe the class label $$c$$ using Bayes' rule as 

$$P(y=c \mid x,\theta) = \frac{P(y=c \mid \theta )P(x \mid y=c,\theta )}{\sum_{z_c}P(y=z_c \mid \theta )P(x \mid y=z_c,\theta )}$$. 

For many applications, of course, we can safely ignore the denominator (e.g. maximazing the probability etc.). The required quantities are the **class-conditional density** $$P(x \mid y=c,\theta)$$ and the **prior** $$P(y=c \mid \theta)$$. In this context, $$ P(y=c \mid x,\theta) $$ is called the **class-posterior**. These models are called generative, because the class-conditional density gives a recipe to generate feature vectors $$x$$. (Directly fitting the class-posterior is called a **discriminative** approach. Note that linear and quadratic discrimnant analysis are despite their names in fact generative models.)

### Naive Bayes

Naive Bayes makes one assumption to simplify the class-conditional distribution dramatically. We assume the features to be independent given the class label (conditionally independent). Now, we can conveniently express the class-conditional distribution in terms of 1-dimensional densities:

$$P(x \mid y=c,\theta) = \prod_{j=1}^{d} P(x_j \mid y=c, \theta_{jc})$$

The reason it is called naive is, of course, that his assumption almost never holds. So, how come it works well for many applications like text classification? Often simplicity, think linear regression, trumps complexity. The low number of parameters (proportional to the number of features $$d$$ times the number of classes) can help protect against overfitting. Another reason is that we can easily deal with different types of features and make additional assumptions about the distributions. 

One common case is document classification (see below) where features are often binary ("is the word present in the document"), i.e. 

$$P(x \mid y=c, \theta) = \prod_{j=1}^d Bernoulli(x_j \mid \theta_{jc}) $$ 

with $$\theta_{jc}$$ being the probability of feature $$j$$ occuring in class $$c$$. Incorporating counts ("how often is the word present in the document") can be achieved easily via the multinomial distribution. It is also straight-forward to extend this to categorical features using the multinoulli distribution. For continous features, a common distribution is the normal distribution with class-proportional means and standard deviations. 

### Training Naive Bayes
I will now show how to compute the maximum likelihood estimate (MLE) for a naive Bayes classifier. 

A single case has the following probability:

$$P(x_i,y_i \mid \theta) = P(y_i \mid \pi) \prod_j{P(x_{ij}\mid \theta_j)}$$

where $\pi$ specifies the class prior. This can be expressed as follows:

$$P(x_i,y_i \mid \theta) = \prod_c{\pi_c^{I(y_i=c)}} \prod_j{\prod_c{P(x_{ij}\mid \theta_{jc})^{I(y_i=c)}}}$$

Thus, the log-likelihood is given by

$$log P(x,y \mid \theta) = \sum_{c} N_c log \pi_c + \sum_j \sum_c \sum_{i:y_i=c} log P(x_{ij} \mid \theta_{jc})$$ 

with $N_c = \sum_i I(y_i=c)$. We can find the maximum using basic Lagrange calculus. It follows easily that the MLE estimate for $\pi_c$ is given by $\frac{N_c}{N}$ (if this is not immediately clear, remember to add the constraint $\sum_c \pi_c =1$ as a Lagrange multiplier). If we assume the features to be conditionally Bernoulli distributed, we have $\theta_{jc} = \frac{N_{jc}}{N_c}$ with $N_{jc} = \sum_i I(x_{ij}=1,y_i=c)$ (the number of samples with feature $j$ in class $c$).

Note that MLE is prone to overfit. In particular, this approach is vulnerable if a feature in the training data is always on. If we encounter a new document where that feature is off, MLE will predice a probability of zero letting the algorithm fail. One solution, of course, is to be "more Bayesian" i.e. use a reasonable prior.

One model that has none of these short-comings is the Dirichlet Compound Multinomial model that fits nicely into the probabilistic modeling framework. 



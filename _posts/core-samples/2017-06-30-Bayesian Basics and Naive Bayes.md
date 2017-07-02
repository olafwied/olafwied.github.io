---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Bayes, prior, posterior, bag of words, document classification]
mathjax: true
---
{% include JB/setup %}

## Bayesian Basics 
The Bayes' rule 

$$
\begin{eqnarray*}P(X=x \mid Y=y) = \frac{ P(X=x, Y=y) }{ P(Y=y)} &=&  
\frac{P(X=x)P(Y=y \mid X=x)}{\sum_{z_x}P(X=z_x)P(Y=y \mid X=z_x)} \end{eqnarray*}
$$

is one of the fundamental equations in probability theory and also in machine learning. One reason is that we can use it to build **generative** classifier as follows: 

Given a feature vector $$x$$, a class label $$c$$ and model described by parameters $$ \theta $$, we can compute the probabilty to observe the class label $$c$$ using Bayes' rule as $$ P(y=c \mid x,\theta) = \frac{P(y=c \mid \theta )P(x \mid y=c,\theta )}{\sum_{z_c}P(y=z_c \mid \theta )P(x \mid y=z_c,\theta )} $$. For most applications, of course, we can safely ignore the denominator (e.g. maximazing the probability etc.). The required quantities are the **class-conditional density** $$P(x \mid y=c,\theta)$$ and the **prior** $$P(y=c \mid \theta)$$. In this context, $$ P(y=c \mid x,\theta) $$ is called the **class-posterior**. These models are called generative, because the class-conditional density gives a recipe to generate feature vectors $$x$$. (Directly fitting the class-posterior is called a **discriminative** approach. Note that linear and quadratic discrimnant analysis are despite their names in fact generative models.)

Naive Bayes makes one assumption to simplify the class-conditional distribution dramatically. We assume the features to be independent given the class label (conditionally independent). Now, we can conveniently express the class-conditional distribution in terms of 1-dimensional densities:

$$P(x \mid y=c,\theta) = \prod_{j=1}^{d} P(x_j \mid y=c, \theta_{jc})

The reason it is called naive is, of course, that his assumption almost never holds. So, how come it works well for many applications like text classification? Often simplicity, think linear regression, trumps complexity. The low number of parameters (proportional to the number of features $$d$$ times the number of classes) can help protect against overfitting. Another reason is that we can easily deal with different types of features and make additional assumptions about the distributions. 

One common case is document classification (see below) where features are often binary (is the word present in the document), i.e. 

$$P(x \mid y=c, \theta) = \prod_{j=1}^d Bernoulli(x_j \mid q_{jc}) $$ with $$q_{jc}$$ being the probability of feature $$j$$ occuring in class $$c$$. Incorporating counts (how often is the word present in the document) can be achieved easily via the multinomial distribution. It is also straight-forward to extend this to categorical features using the multinoulli distribution. For continous features, a common a distribution is the normal distribution with class-proportional means and standard deviations. 
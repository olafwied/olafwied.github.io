---
layout: post
category : Machine Learning Basics
tagline: hands-on
tags : [Bayes, prior, posterior, bag of words, document classification]
mathjax: true
---
{% include JB/setup %}

##Bayesian Basics 

The Bayes' rule 

$$
\begin{eqnarray*}P(X=x | Y=y) = \frac{ P(X=x, Y=y) }{ P(Y=y)} &=&  
\frac{P(X=x)P(Y=y | X=x)}{\sum_{z_x}P(X-z_x)P(Y=y | X=z_x)} \end{eqnarray*}
$$

is one of the fundamental equations in probability theory and also in machine learning. One reason is that we can use it to build **generative** classifier as follows: 

Given a feature vector $$x$$, a class label $$c$$ and model described by parameters $$\theta$$, we can compute the probabilty to observe the class label $$c$$ using Bayes' rule as $$P(y=c | x,\theta) = \frac{P(y=c|\theta)P(x|y=c,\theta)}{sum_{z_c}P(y=z_c|\theta)P(x | y=z_c,\theta)}$$. For most applications, of course, we can safely ignore the denominator (e.g. maximazing the probability etc.). The required quantities are the **class-conditional density** $$P(x|y=c,\theta)$$ and the **prior** $$P(y=c|\theta)$$. In this context, $$P(y=c | x,\theta)$$ is called the **class-posterior**. These models are called generative, because the class-conditional density gives a recipe to generate feature vectors $$x$$. (Directly fitting the class-posterior is called a **discriminative** approach. Note that linear and quadratic discrimnant analysis are despite their names in fact generative models.)

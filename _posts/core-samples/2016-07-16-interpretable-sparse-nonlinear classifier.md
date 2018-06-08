---
layout: post
category : Machine Learning
tagline: hands-on
tags : [Sparse, Non-linear, interpretable, Subspace, KDE, FFD]
mathjax: true
---
{% include JB/setup %}

## Introduction

With this post, I'd like to introduce the Fast Flux Discriminant (FFD) classifier to a wider audience. FFD is a learning method developed by researchers at the University of Washington in 2014. 
It has some interesting properties in that it can model efficiently model non-linear feature interactions (with some limitations), allows for sparse solutions and is easy to interpret. Easy interpretation is often lacking for non-linear classification methods.  
It can be seen as a generalization of the linear ($L_1$-regularized) logistic regression. 

## Basics

FFD is based on (non-parametric) kernel density estimation (KDE) in selected sub-spaces. KDE tries to directly estimate the density of $P(y \mid x)$, leading to a discriminative classifier. (KDE basically works by dividing the input space into a grid and do a histogram count with some appropriate smoothing.) However, KDE quickly becomes infeasible for high-dimensional data, but can be very efficient in low-dimensional spaces. Following [Chen, W. et al. 2014], let $D$ be the number of features and $r < D$ be the maximal number of interactions between features. 
The main idea of FFD is the following: 
 
1. Select a set of features $A$, with $\mid A \mid \leq r$ and corresponding ("shortened") input vector $x_A$ 
2. Estimate $P(y\mid x_A)$ using KDE 
3. Learn a (non-linear) feature $z \mapsto \Phi(z)$ 
4. Learn a sparse, linear classifier on the data $\Phi(x)$

In particular, $\Phi(x) = \left(\phi_1(x),\ldots,\phi_m(x)\right)$ with 

$$\phi_m(x) = \log \left(\frac{P(y=1 \mid x_A)}{P(y\neq 1 \mid x_A)}\right).$$

$P(y\mid x_A)$ is estimated as the proportion of training samples with $y=1$ in each grid cell of the KDE. Further, the last step is lgositic regression with the advantage that transformed features are all positive. Thus, $L_1$-regularization can be re-formulated to a smooth (differntiable) optimization problem that can be solved very efficiently. (See section 3.4 for details.)
We can see why FFD is more interpretable than other non-linear classifier. The weights of the linear classifer learned on the transformed data provide insight into the importance of the feature bags. If the bags are small, we can easily identify relevant feature interactions.

## Complexity

One difficulty is the large number of possible feature bags $A$, namely $\binom{D}{r}$. FFD tackles this problem by formulating a combinatorial optimization problem based on the Pearson correlation between transformed features. 

## Drawbacks

So, why isn't the FFD algorithm more widely adopted? First, in practice, $L_1$ usually underperforms compared to $L_2$ regularization and is often only desireable for specific problems (for example in biogenetics) or for ease of interpretation. In addition, the number of interactions $r$ has to be small to control the computational complexity. Also, the full algorithm hasn't been implemented in one of the popular machine learning libraries outside Matlab. However, there is an efficient C algorithm for the sub-space selection of which I have built a simplified version of the algorithm with Cython, which allows for easy integration between Python and C. For complex problems, standard (black-box) kernel methods like SVM seem to outperform FFD. But, for cetain applications, in particular in business and medicine where interpretable results are often critical, this is a very interesting solution! Try it out and let me know how it works for you.

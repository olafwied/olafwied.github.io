---
layout: post
category : Machine Learning
tagline: hands-on
tags : [Gaussian Mixture Model, Bayes, Latent Variables]
mathjax: true
---
{% include JB/setup %}

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
```

## Introduction to the Expectation-Maximization Algorithm
### The Gaussian Mixture Model (GMM)

The Gaussian Mixture Model is quite simple. We assume the data is generated from a mixture of Gaussian variables, namely:

$P(x|\theta) = w_1 \cdot N(x|\mu_1, \Sigma_1) + \ldots + w_K \cdot N(x|\mu_K, \Sigma_K)$ where $\theta$ is the set of all parameters, in this case the location vectors and covariance matrices $\{\mu_1,\ldots,\mu_K,\Sigma_1,\ldots,\Sigma_K\}$ and $K$ is the number of "mixtures".

Note that $\sum_{j=1}^K w_j = 1$ and $w_j \geq 0 \,, \forall j=1,\ldots,K$ to obtain a true probability distribution. 

The data is distributed as a weighted average of Gaussian variables. Let's see an example for $K=2$.


```python
np.random.seed = 123


n = 30
mu1 = -2.0
mu2 = 1.0
sigma1 = 1.3
sigma2 = 0.9
cluster_1 = np.random.normal(mu1, sigma1, n)
cluster_2 = np.random.normal(mu2, sigma2, n)
def norm_density(x, mu=0.0, sigma=1.0):
    c = np.sqrt(2.0*np.pi*sigma**2)
    return np.exp(-np.square(x-mu)/(2*sigma**2))/c
```


```python
x = np.linspace(-6,6,500)
jitter1 = np.random.rand(n)*0.02
jitter2 = np.random.rand(n)*0.02
#plot densities
plt.plot(x, norm_density(x,mu1,sigma1),'b-')
plt.plot(x, norm_density(x,mu2,sigma2),'r-')
#plot samples
plt.plot(cluster_1,jitter1,'b*',label='C=1')
plt.plot(cluster_2,jitter2,'r*',label='C=2')
plt.legend()
plt.show()
```


![png](olafwied.github.io/_posts/output_3_0[1].png)


If we wanted to maximize the likelihood $\prod P(x_j|\theta)$ (subject to the positivity constraints on $w$), we could simply use a optimazation rountine like stochastic gradient descent. However, this can converge slowly and often to suboptimal solutions. 

The Expectation-Maximazation Algorithm (or EM) can often converge after a few steps to a nearly optimal solution.

### The Idea behind the EM Algorithm

To understand the EM algorithm, we need to introduce a latent (unobserved) variable. This will simplify the calculations as we will see shortly.

Let's a latent variable $T$ that "assigns" our data to one of the clusters.

1. $P(T=c|\theta) = w_c$
2. $P(x|T=c, \theta) = N(x|\mu_c, \Sigma_c)$

That is, given the assigned cluster $c$ by $T$, $x$ follows the normal distribution defined by the corresponding parameters $\mu_c$ and $\Sigma_c$.

How does this help? 

#### What if we knew $T$?

Now, if we know $T$, we can estimate the parameters $\theta$. If we know $T$, we know probabilies that x is of color $c$ for all samples and therefore could estimate the parameters of the Gaussians. In our example above:

$P(x \,|\, T=c, \theta) = N(x \,|\ \mu_c, \sigma^2_c)$ and from here

- $\mu_c = \frac{\sum_{j}P(T=c \,|\ x_j, \theta)\cdot x_j}{\sum_j P(T=c \,|\ x_j, \theta)}$, 
- $\sigma^2_1 = \frac{\sum_{j}P(T=c \,|\ x_j, \theta)\cdot (x_j- \mu_c)^2}{\sum_j P(T=c \,|\ x_j, \theta)}$

and similarly for the red points for $T=2$. 

Note that this gives us "soft assignments" where each sample is not assigned just one (the most likely) cluster but a probability for each cluster $P(T=c \,|\ x_j, \theta)$. For "hard clustering" we take the average over all blue or red points to compute the parameters of the Gaussians.


#### What if we knew $\theta$?

On the other hand, if we know $\theta$ we can compute the distribution of $T$ just as easilty:

- $P(T=c \,|\, x, \theta) = const \cdot P(x \,|\, T=c, \theta)\cdot P(T=c\,|\,\theta)$, which is simply the Bayes rule with some normalizing constant.

#### EM Algorithm as Chicken and Egg Problem
So far, we have seen that if we know one of the two, we can estimate the other. Therefore, we are left with a chicken and egg problem. Where to start? 

The idea of the EM algorithm is simply and powerful. Start with a random initialization and then estimate the parameters. In the next step, take the parameters as fixed and estimate the latent variable. Now, fix the latent variable distribution and estimate the parameters again with the updated values. Continue this iteration between the so-called "expectation" and "maximazation" steps until convergence.

Let's try this for our example. Let's start by choosing two initial Gaussians:

#### Initialization


```python
mu1_init = -0.5
mu2_init = 0.5
sigma1_init = 1.0
sigma2_init = 1.0

#plot densities
plt.plot(x, norm_density(x,mu1_init,sigma1_init),'b-',label='C=1')
plt.plot(x, norm_density(x,mu2_init,sigma2_init),'r-',label='C=2')
#plot samples
plt.plot(cluster_1,jitter1,'k*')
plt.plot(cluster_2,jitter2,'k*')
plt.legend()
plt.title("Initial Distributions")
plt.show()
```


![png](olafwied.github.io/_posts/output_11_0[1].png)


#### Infer Assignments

Now infer the soft assignments from the initial distributions. We can ignore the normalizing constant because we don't need true probabilities to do our soft assignments!

1. $P(T=1 \,|\, x, \theta) = P(x \,|\, T=1, \theta)\cdot P(T=1 \,|\,\theta) = N(x \,|\ -0.5, 1.0) \cdot w_1$
2. $P(T=2 \,|\, x, \theta) = P(x \,|\, T=2, \theta)\cdot P(T=2 \,|\,\theta) = N(x \,|\ 0.5, 1.0) \cdot w_2 = N(x \,|\ 0.5, 1.0) \cdot (1-w_1)$

Initially, we assume $w_1=w_2=0.5$.


```python
cluster1_c1 = norm_density(cluster_1, mu1_init, 0.5**2)
cluster2_c1 = norm_density(cluster_2, mu1_init, 0.5**2)
```


```python
cluster1_c2 = norm_density(cluster_1, mu2_init, 0.5**2)
cluster2_c2 = norm_density(cluster_2, mu2_init, 0.5**2)
```


```python
assignments_cluster1 = np.argmax(np.vstack([cluster1_c1,cluster1_c2]).T, axis=1)
```


```python
assignments_cluster2 = np.argmax(np.vstack([cluster2_c1,cluster2_c2]).T, axis=1)
```


```python
#plot densities
plt.plot(x, norm_density(x,mu1_init,sigma1_init),'b-',label='C=1')
plt.plot(x, norm_density(x,mu2_init,sigma2_init),'r-',label='C=2')
#plot samples
plt.scatter(cluster_1,jitter1,marker='*',c=['b' if x==0 else 'r' for x in assignments_cluster1])
plt.scatter(cluster_2,jitter2,marker='*',c=['b' if x==0 else 'r' for x in assignments_cluster2])
plt.legend()
plt.title("Initial Assignments")
plt.show()
```


![png](olafwied.github.io/_posts/output_18_0[1].png)


#### Update Parameters

Now, let's estimate the parameters of the Gaussians based on the initial assignments:


```python
cluster_1, assignments_cluster1
```




    (array([-2.93218598, -3.49615562, -0.30486232, -1.40113041, -1.7226102 ,
            -0.63181531, -2.04186532, -1.80173694, -2.7921006 , -1.0785547 ,
            -2.88087361, -2.81908622, -0.98187877, -2.482527  , -0.45299344,
            -2.91480084, -4.69573539, -1.16825533, -2.79565947, -0.69627068,
            -0.85803205, -3.47278294,  0.42156497, -3.7212476 , -1.73476679,
            -1.54731854, -1.13937859, -2.61531047, -2.49307557, -2.38927165]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0], dtype=int64))




```python
cluster_2, assignments_cluster2
```




    (array([ 2.00764816,  0.61564616,  1.53224356,  0.05894657,  0.85670316,
             0.53587344,  0.79208366,  2.34972749, -0.53837314,  2.01461064,
             1.17906862,  1.36325329,  0.44832125,  2.86156194,  1.23458584,
             1.2745659 ,  2.49968063,  1.04963874,  1.52950547,  0.82285498,
             0.91453704,  2.03576486,  2.2033406 , -0.06586041,  0.61639736,
             2.08018301,  2.09991788,  1.95452593, -0.46573425,  0.91396823]),
     array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 0, 1], dtype=int64))




```python
mu1_est = np.sum(cluster_1[np.where(assignments_cluster1==0)])+np.sum(cluster_2[np.where(assignments_cluster2==0)])
mu1_est /= np.sum(assignments_cluster1==0)+np.sum(assignments_cluster2==0)
```


```python
mu2_est = np.sum(cluster_1[np.where(assignments_cluster1==1)])+np.sum(cluster_2[np.where(assignments_cluster2==1)])
mu2_est /= np.sum(assignments_cluster1==1)+np.sum(assignments_cluster2==1)
```


```python
sigma1_est = np.sum(np.square(cluster_1[np.where(assignments_cluster1==0)]-mu1_est))+np.sum(np.square(cluster_2[np.where(assignments_cluster2==0)]-mu1_est))
sigma1_est /= np.sum(assignments_cluster1==0)+np.sum(assignments_cluster2==0)
```


```python
sigma2_est = np.sum(np.square(cluster_1[np.where(assignments_cluster1==1)]-mu2_est))+np.sum(np.square(cluster_2[np.where(assignments_cluster2==1)]-mu2_est))
sigma2_est /= np.sum(assignments_cluster1==1)+np.sum(assignments_cluster2==1)
```


```python
mu1_est, sigma1_est, mu2_est, sigma2_est
```




    (-1.9103828171003543,
     1.3201003428930016,
     1.3666685484208816,
     0.52251126923143254)



And, let's plot the updated distributions:


```python
#plot densities
plt.plot(x, norm_density(x,mu1_est,sigma1_est),'b-',label='C=1')
plt.plot(x, norm_density(x,mu2_est,sigma2_est),'r-',label='C=2')
#plot samples
plt.scatter(cluster_1,jitter1, marker='*',c='k')
plt.scatter(cluster_2,jitter2, marker='*', c='k')
plt.legend()
plt.title("Updated Parameters")
plt.show()
```


![png](olafwied.github.io/_posts/output_29_0[1].png)


**We can see that already after one step, we are pretty close to the true parameters of the Gaussian distributions!**

#### Infer Assignments (Round 2)


```python
cluster1_c1 = norm_density(cluster_1, mu1_est, 0.5**2)
cluster2_c1 = norm_density(cluster_2, mu1_est, 0.5**2)
```


```python
cluster1_c2 = norm_density(cluster_1, mu2_est, 0.5**2)
cluster2_c2 = norm_density(cluster_2, mu2_est, 0.5**2)
```


```python
assignments_cluster1 = np.argmax(np.vstack([cluster1_c1,cluster1_c2]).T, axis=1)
```


```python
assignments_cluster2 = np.argmax(np.vstack([cluster2_c1,cluster2_c2]).T, axis=1)
```


```python
#plot densities
plt.plot(x, norm_density(x,mu1_est,sigma1_est),'b-',label='C=1')
plt.plot(x, norm_density(x,mu2_est,sigma2_est),'r-',label='C=2')
#plot samples
plt.scatter(cluster_1,jitter1,marker='*',c=['b' if x==0 else 'r' for x in assignments_cluster1])
plt.scatter(cluster_2,jitter2,marker='*',c=['b' if x==0 else 'r' for x in assignments_cluster2])
plt.legend()
plt.title("Updated Assignments")
plt.show()
```


![png](olafwied.github.io/_posts/output_36_0[1].png)


#### Update Parameters (Round 2)


```python
mu1_est = np.sum(cluster_1[np.where(assignments_cluster1==0)])+np.sum(cluster_2[np.where(assignments_cluster2==0)])
mu1_est /= np.sum(assignments_cluster1==0)+np.sum(assignments_cluster2==0)
```


```python
mu2_est = np.sum(cluster_1[np.where(assignments_cluster1==1)])+np.sum(cluster_2[np.where(assignments_cluster2==1)])
mu2_est /= np.sum(assignments_cluster1==1)+np.sum(assignments_cluster2==1)
```


```python
sigma1_est = np.sum(np.square(cluster_1[np.where(assignments_cluster1==0)]-mu1_est))+np.sum(np.square(cluster_2[np.where(assignments_cluster2==0)]-mu1_est))
sigma1_est /= np.sum(assignments_cluster1==0)+np.sum(assignments_cluster2==0)
```


```python
mu1_est = np.sum(cluster_1[np.where(assignments_cluster1==0)])+np.sum(cluster_2[np.where(assignments_cluster2==0)])
mu1_est /= np.sum(assignments_cluster1==0)+np.sum(assignments_cluster2==0)
```


```python
mu1_est, sigma1_est, mu2_est, sigma2_est
```




    (-1.9698835399506125,
     1.2493934722913047,
     1.3172709981738739,
     0.52251126923143254)




```python
#plot densities
plt.plot(x, norm_density(x,mu1_est,sigma1_est),'b-',label='C=1')
plt.plot(x, norm_density(x,mu2_est,sigma2_est),'r-',label='C=2')
#plot samples
plt.scatter(cluster_1,jitter1,marker='*',c=['b' if x==0 else 'r' for x in assignments_cluster1])
plt.scatter(cluster_2,jitter2,marker='*',c=['b' if x==0 else 'r' for x in assignments_cluster2])
plt.legend()
plt.title("Updated Parameters")
plt.show()
```


![png](olafwied.github.io/_posts/output_43_0[1].pngg)


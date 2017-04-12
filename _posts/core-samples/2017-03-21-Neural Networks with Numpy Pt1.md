---
layout: post
category : Neural Networks
tagline: hands-on
tags : [Neural Network, Deep Learning, Python, feedforward, backpropagation, mini-batch]
---
{% include JB/setup %}

What is this series about?
==========================

This is my attempt to explain deep learning to myself (sorry guys). It is based on my belief that if you can't implement it, you don't understand it. There is a lot I thought I understood before I started this series. I was often wrong. 

With this in mind, it should be clear that this is not supposed to be another tutorial on deep learning. At the end of the blog, Google will probably not hire you (or me). But, you might have a better and clearer mental picture of neural networks that can make you more productive and hands-on.

Part 1
======

In part 1, we will implement a deep feedforward network. (Surprise, right?)
Feedforward networks are the conceptual building blocks most types of neural networks. A thourough understanding of feedforward networks is a critical first step. The main goal is to understand backpropagation. A second goal is a better understanding of the architecture of feedforward networks. We will achieve this by implementing them in such a way that the code mirrors the general ideas of a feedforward architecture. In particular the dimensionality of all the components. This will allow us to set up arbitrary deep and wide networks, different non-linearities in the hidden units, as well as different output and loss functions. On the way, this exercise will force us to think about the dimensionality of all the components of the network and how they interact.

Feedforward networks consist of an input layer, one or more hidden layers and an output layer. All layers are fully connected, but the connections only go in the "forward" direction from inputs to outputs.

The implementation should allow to easily swap out hidden layers with different non-linearities (ReLu, Sigmoid etc.). Same for the outputlayer. The implementation should be flexible to handle regression and classification output layers with different loss functions. The implementation should allow training with stochastic gradient descent with backpropagation and mini-batch. However,to get this started, we use the following example as an orientation:

Input Layer - Fully Connected Layer with ReLu - Fully Connected Layer with ReLu - Softmax Output with Negative Log Likelihood Loss (Maximum Likelihood)

with a mini-batch size of 1 (online SGD). That is a classification setting with two hidden ReLu layers. 

Throughout this series, I'll try to explain my reasoning why I approached problems a certain way. The code will go through many iterations. Note that none of the implementations are aimed at efficieny but at a conceptual understanding of the most important concepts.

Once succesful, we will generalize this approach to mini-batches of size m>1.

What we will not cover
---------------------- 
To complete the absolute basics, we will address how to handle parameters more elegantly and regularization in part 2.
There are tons of variations and adjustments that will not be covered in the first two 2 posts. To give you and idea of how little we are really covering, here are some of things that we will not have covered:
- Advanced regularization like dropout (part 3)
- Convolutional and Recurrent nets (part4?)
- Other optimization routines (AdamGrad, Momentum,...)
- Batch normalization
- Modern graph implementations of gradient descent (think Tensorflow, Theano etc.)
- Inputs other than vectors (tensors)

The last two will not be covered in later parts.

Basics
------

Let's start by implementing the example with two hidden layers in a modular fashion. 

A natural first step is to implement a neural network class which consists of multiple layers. 

  A layer is defined by its  number of input and output units, its weights and biases and its activation function. 
    In a fully connected layer, the number of input units is equal to the number of output units in the previous layer. The initialization of our network class has to take this into account.
  The weights and biases of each layer need to be initialized appropriately. 
To train the net we need a function that implements the forward sweep, the backward sweep and the gradient updates to the weights. 

The forward sweep is simple. Take the inputs and compute the affine transformation with the current weights and bias. Then apply the non-linearity and pass the result to the next layer. Do this for every layer until you reach the outout layer. Clearly, it makes sense that the forward function of the network iteratively calls the forward functions of every layer.

The backpropagation performs a similar procedure in the opposite directions on the activations. Here is a quick recap of basic backpropagation. We will give a quick recap at the end of this section.

Alright, let's get down to business. But wait! Before we can start implementing stuff, we need to be clear about the dimensions of all the variables:

We look at the network as moving top down instead of left right. Every layer is hence represented by a row vector. Our inputs will be row vectors as well. We do so, because when we generalize our online approach to more than one sample, we would like the input to be a design matrix where each row represents one sample. Unfortunately, this yields results that look a bit different from traditional literature. But it felt like a natural starting point. We will reiterate later if necessary. 
The weight connections are represented by a matrix $W = (w_{ij})$, where the entry $w_ij$ connects the neuron $i$ of the previous layer with neuron $j$ of the current layer. We use superscripts to indicate the layers. We call the input $X$ or $h^{0}. We then define, for $l=1,...L$, $a^l = h^{l-1} W^l + b^l$ as the input to the activation function $f^l$ and $h^l = f^l(a^l)$ as the output. Let $n_l$ be the number of neurons in every layer. The weight connections then have dimensions $n_{l-1} \times n_l$. The loss function $J$ takes as input $h^L$ and the labels $y$ and outputs a scalar.

To recap, $h^l$, $a^l$, $b^l$ are all row vectors. $W^l$ is a matrix with as many rows as neurons in layer $l-1$ and as many columns as neurons in layer $l$. In the hidden layers, $f^l$ is a non-linear function that is applied elementwise. $y$ is either a scalar (e.g. standard regression) or a row-vector (e.g. classification with $n_L$ classes; here $y$ is a vector of zeros with a single 1 that indicates the true class label). $J$ is the loss function mapping $h^L$ and $y$ to a scalar loss value.

Our primary goal is now to compute the gradients of $J$ with respect to the weights $W$ and biases $b$. Then, we can update the weights using gradient descent. As mentioned over and over again everywhere, backpropoagation is simply the repeated use of the chain rule. The trick to see how this can be done in a efficient and iterative fashion is to compute the gradients with respect to the activation inputs $a$ instead of the outputs $h$. Let's start with $\frac{\partial J}{\partial w^L_{ij}} = \sum_{k=1}^{n_L} \frac{\partial J}{\partial a^L_{k}} \frac{\partial a^L_{k}}{\partial w^L_{ij}}$.

It is easy to see from the definition of $a$, that $a^j_k$ only depends on $$w^j_{ij}$$ if \$$k=j$$. Therefore, the expression simplifies to $\frac{\partial J}{\partial a^L_{j}} \frac{\partial a^L_{j}}{\partial w^L_{ij}}$. Next, since $a$ is linear in $w$, we get $\frac{\partial J}{\partial a^L_{k}} h^{L-1}_{i}$. In vector form we get, the Jacobian of $L$ w.r.t. $w^L$ is given by the outer product $(h^{L-1})^T \nabla_{a^L}L$. Let's do one more step to derive the backpropagation algorithm. Similarly, we obtain
$\frac{\partial J}{\partial a^{l-1}_j} = 
\sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial a^{l-1}_j} =
\sum_{k=1}^{n_{l}}\sum_{i=1}^{n_{l-1}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial h^{l-1}_i} \frac{\partial h^{l-1}_i}{\partial a^{l-1}_j} = 
\sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial h^{l-1}_j} \frac{\partial h^{l-1}_j}{\partial a^{l-1}_j} = 
\sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial h^{l-1}_j} f^{l-1}{'}(a^{l-1}_j) = 
\sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} w^l_{jk} f^{l-1}{'}(a^{l-1}_j)$

In vector form, this looks as follows: $\frac{\partial J}{\partial a^{l-1}} = \nabla_{a^{l}}J \, {W^{l}}^{T} \circ f{'}(a^{l-1})$.

From here, it is a small step to $\frac{\partial J}{\partial w^{l-1}_{ij}} = \sum_{k=1}^{n_{l-1}} \frac{\partial J}{\partial a^{l-1}_{k}} \frac{\partial a^{l-1}_{k}}{\partial w^{l-1}_{ij}} = \frac{\partial J}{\partial a^{l-1}_{j}} \frac{\partial a^{l-1}_{j}}{\partial w^{l-1}_{ij}} = \frac{\partial J}{\partial a^{l-1}_{j}} h^{l-2}_i$.

Again, in vector form: ${h^{l-2}}^{T} \nabla_{a^{l-1}}J$.

Now, the recipe is clear. We propagate the gradient of $J$ w.r.t. to $a$ backwards using the weights $W$. We perform a vector product with $h$ of the previous layer (the layer closer to the output since we are moving backwards) to obtain the gradient w.r.t. to the weights. 

Algorithm
---------

In pseudo code, we get the following basic backpropagation algorithm:


Implementation
--------------

A forward sweep updates the variables $a$ and $h$. Since $a^l$ depends on $h^{l-1}$, we will implement the forward method such that it takes $h^{l-1}$ as input. Note that this basic implementation will keep all the variables $W, b, a, h$ in memory. This makes sense also for $a$ and $h$ because they play their part in the backpropagation algorithm.

A backward sweep updates the variables $g, \nabla_b J$ and $\nabla_W J$. Since $g$ is not part of the forward propagation, there is no point in keeping it in memory for every layer. 


---backprop gradient with original $W$ to propagate the correct error of the forward step
---show example how loss decrease when running the same sample
Of course, this example is silly but hopefully convinces you that we are not completely wrong so far.
... 

---now mini-batch
---have to average to update $b$ and divide throug $m$ to update $W$, otherwise the same because of our row-approach
---show example how loss decrease when running the same mini-batch

---experiment comparing sgd and mini-batch
Part 2
======

So far, the parameters (like the learning rate and the initial values of $W$ and $b$) are hand-coded into the definition of the classes. This is unsatsifactory. For example, we would like to specify the initialization scheme when we define our network architecture. However, the learning is not really part of our network architecture. It seems to be tied closer to the learning procedure. Therefore, it makes to sense to separate the learning algorithm from the network architecture. 
To not only deal with code reorganization, we will also introduce how to easily incorporate well-known regularization into our gradient descent update rules. The math is very basic, so this part just be easy to follow.

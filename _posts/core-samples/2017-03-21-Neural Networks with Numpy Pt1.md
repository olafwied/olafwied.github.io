---
layout: post
category : Neural Networks
tagline: hands-on
tags : [Neural Network, Deep Learning, Python, feedforward, backpropagation, mini-batch]
mathjax: true
---
{% include JB/setup %}

... in progress ...

## What is this series about?


This is my attempt to explain deep learning to myself hands-on. It is based on my belief that if you can't implement it, you don't understand it. There is a lot I thought I understood before I started this series. I was often wrong. 

With this in mind, it should be clear that this is not supposed to be another tutorial on deep learning. At the end of the blog, Google will probably not hire you (or me). But, you might have a greater appreciation of all the modern, high-performance libraries out there and maybe also clearer mental picture of neural networks that can make you more productive and hands-on.

## Part 1

In part 1, we will implement a deep feedforward network. (Surprise, right?)
Feedforward networks are the conceptual building blocks most types of neural networks. A thourough understanding of feedforward networks is a critical first step. The main goal is to understand backpropagation. A second goal is a better understanding of the architecture of feedforward networks. We will achieve this by implementing them in such a way that the code mirrors the general ideas of a feedforward architecture. In particular, the dimensionality of all the components. This will allow us to set up arbitrary deep and wide networks, different non-linearities in the hidden units, as well as different output and loss functions. On the way, this exercise will force us to think about the dimensionality of all the components of the network and how they interact.

Feedforward networks consist of an input layer, one or more hidden layers and an output layer. All layers are fully connected, but the connections only go in the "forward" direction from the inputs towards the output(s).

The implementation should allow to easily swap out hidden layers with different non-linearities (ReLu, Sigmoid etc.). Same for the outputlayer. The implementation should be flexible to handle regression and classification output layers with different loss functions. The implementation should allow training with stochastic gradient descent with backpropagation and mini-batch. However,to get this started, we use the following example as an orientation:

Input Layer - Fully Connected Layer with ReLu - Fully Connected Layer with ReLu - Softmax Output with Negative Log Likelihood Loss (Maximum Likelihood)

That is a classification setting with two hidden ReLu layers. 

Throughout this series, I'll try to explain my reasoning why I approached problems a certain way. The code will go through many iterations. Note that none of the implementations are aimed at efficieny but at a conceptual understanding of the most important concepts.

Once succesful, we will generalize this approach to more advanced concepts.

### What we will not cover 
To complete the absolute basics, we will address how to handle parameters more elegantly and regularization in part 2.
There are tons of variations and adjustments that will not be covered in the first two 2 posts. To give you and idea of how little we are really covering, here are some of things that we will not have covered:
- Advanced regularization like dropout (part 3?)
- Convolutional and Recurrent nets (part4?)
- Other optimization routines (AdamGrad, Momentum,...)
- Batch normalization
- Modern graph implementations of gradient descent (think Tensorflow, Theano etc.)
- Inputs other than vectors (tensors)

The last two will not be covered in later parts.

### Basics

Let's start by implementing the example with two hidden layers in a modular fashion. 

A natural first step is to implement a neural network class which consists of multiple layers. 

  A layer is defined by its  number of input and output units, its weights and biases and its activation function. In a fully connected layer, the number of input units is equal to the number of output units in the previous layer. The initialization of our network class has to take this into account. The weights and biases of each layer need to be initialized appropriately. 
To train the net, we need a function that implements the forward sweep, the backward sweep and the gradient updates to the weights. 

The forward sweep is simple: Take the inputs and compute the affine transformation with the current weights and bias. Then, apply the non-linearity and pass the result to the next layer. Do this for every layer until you reach the outout layer. Clearly, it makes sense that the forward function of the network iteratively calls the forward functions of every layer.

The backpropagation performs a similar procedure in the opposite directions on the activations. We will give a quick recap at the end of this section.

Alright, let's get down to business. But wait! Before we can start implementing stuff, we need to be clear about the dimensions of all the variables:

We look at the network as moving top down instead of left right. Every layer is hence represented by a row vector. Our inputs will be row vectors as well. We do so, because when we generalize our online approach to more than one sample, we would like the input to be a design matrix where each row represents one sample. Unfortunately, this yields results that look a bit different from traditional literature. But it felt like a natural starting point. We will reiterate later if necessary. 
The weight connections are represented by a matrix $$W = (w_{ij})$$, where the entry $$w_{ij}$$ connects the neuron $$i$$ of the previous layer with neuron $$j$$ of the current layer. We use superscripts to indicate the layers. We call the input $$X$$ or $$h^{0}$$. We then define, for $$l=1,...L$$, $$a^l = h^{l-1} W^l + b^l$$ as the input to the activation function $$f^l$$ and $$h^l = f^l(a^l)$$ as the output. Let $$n_l$$ be the number of neurons in every layer. The weight connections then have dimensions $$n_{l-1} \times n_l$$. The loss function $$J$$ takes as input $$h^L$$ and the labels $$y$$ and outputs a scalar.

To recap, $$h^l$$, $$a^l$$, $$b^l$$ are all row vectors. $$W^l$$ is a matrix with as many rows as neurons in layer $$l-1$$ and as many columns as neurons in layer $$l$$. In the hidden layers, $$f^l$$ is a non-linear function that is applied elementwise. $$y$$ is either a scalar (e.g. standard regression) or a row-vector (e.g. classification with $$n_L$$ classes; here $$y$$ is a vector of zeros with a single 1 that indicates the true class label). $$J$$ is the loss function mapping $$h^L$$ and $$y$$ to a scalar loss value.

Our primary goal is now to compute the gradients of $$J$$ with respect to the weights $$W$$ and biases $$b$$. Then, we can update the weights using gradient descent. As mentioned over and over again everywhere, backpropoagation is simply the repeated use of the chain rule. The trick to see how this can be done in a efficient and iterative fashion is to compute the gradients with respect to the activation inputs $$a$$ instead of the outputs $$h$$. Let's start with $$\frac{\partial J}{\partial w^L_{ij}} = \sum_{k=1}^{n_L} \frac{\partial J}{\partial a^L_{k}} \frac{\partial a^L_{k}}{\partial w^L_{ij}}$$.

It is easy to see from the definition of $$a$$, that $$a^j_k$$ only depends on $$w^j_{ij}$$ if $$k=j$$. Therefore, the expression simplifies to $$\frac{\partial J}{\partial a^L_{j}} \frac{\partial a^L_{j}}{\partial w^L_{ij}}$$. Next, since $$a$$ is linear in $$w$$, we get $$\frac{\partial J}{\partial a^L_{k}} h^{L-1}_{i}$$. In vector form we get, the Jacobian of $$J$$ w.r.t. $$w^L$$ is given by the outer product $$(h^{L-1})^T \nabla_{a^L}J$$. Let's do one more step to derive the backpropagation algorithm. Similarly, we obtain
$$\begin{eqnarray*}\frac{\partial J}{\partial a^{l-1}_j} &=& 
\sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial a^{l-1}_j} =
\sum_{k=1}^{n_{l}}\sum_{i=1}^{n_{l-1}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial h^{l-1}_i} \frac{\partial h^{l-1}_i}{\partial a^{l-1}_j} \\\\\\
&=& \sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial h^{l-1}_j} \frac{\partial h^{l-1}_j}{\partial a^{l-1}_j} = 
\sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} \frac{\partial a^{l}_k}{\partial h^{l-1}_j} f^{l-1}{'}(a^{l-1}_j) \\\\\\
&=& \sum_{k=1}^{n_{l}} \frac{\partial J}{\partial a^{l}_k} w^l_{jk} f^{l-1}{'}(a^{l-1}_j) \end{eqnarray*}$$

In vector form, this looks as follows: $$\frac{\partial J}{\partial a^{l-1}} = \nabla_{a^{l}}J \, {W^{l}}^{T} \circ f{'}(a^{l-1})$$.

From here, it is a small step to $$\frac{\partial J}{\partial w^{l-1}_{ij}} = \sum_{k=1}^{n_{l-1}} \frac{\partial J}{\partial a^{l-1}_{k}} \frac{\partial a^{l-1}_{k}}{\partial w^{l-1}_{ij}} = \frac{\partial J}{\partial a^{l-1}_{j}} \frac{\partial a^{l-1}_{j}}{\partial w^{l-1}_{ij}} = \frac{\partial J}{\partial a^{l-1}_{j}} h^{l-2}_i$$.

Again, in vector form: $${h^{l-2}}^{T} \nabla_{a^{l-1}}J$$.

Now, the recipe is clear. We propagate the gradient of $$J$$ w.r.t. to $$a$$ backwards using the weights $$W$$. We perform a vector product with $$h$$ of the previous layer (the layer closer to the output since we are moving backwards) to obtain the gradient w.r.t. to the weights. 

### Algorithm

In pseudo code, we get the following basic backpropagation algorithm:


### Implementation

The only package we rely on for our implementation is numpy:
```python
import numpy as np
import matplotlib.pyplot as plt
```

A forward sweep updates the variables $$a$$ and $$h$$. Since $$a^l$$ depends on $$h^{l-1}$$, we will implement the forward method such that it takes $$h^{l-1}$$ as input. Note that this basic implementation will keep all the variables $$W, b, a, h$$ in memory. This makes sense also for $$a$$ and $$h$$ because they play their part in the backpropagation algorithm. A typical forward step looks something like that
```Python
a = np.dot(h_prev_layer,self.w)+self.b
```

A backward sweep updates the variables $$g, \nabla_b J$$ and $$\nabla_W J$$. Since $$g$$ is not part of the forward propagation, there is no point in keeping it in memory for every layer. 
```python%start_inline=true
g = g*f_prime
g_back = np.dot(g,self.w.T)
```

Let's piece it together, starting with our Neural Network class, that takes in a list of layers.
```python%start_inline=true
class Net(object):
    """
    Parameters
    ----------
    layers: list of dicts
        a list of dicts specifying the layers, see documentation for more details
    
    """
    def __init__(self,layers):
        self.n_layers = len(layers)
        if layers[0]['layer_type'] != 'input':
            raise ValueError('First Layer needs to be an input layer')
        if layers[-1]['layer_type'] != 'output':
            raise ValueError('Last layer needs to be an output layer')
        self.layers = [InputLayer(layers[0]['batch_size'],layers[0]['n_features'])]
        for j in range(1,self.n_layers-1):
            if layers[j]['layer_type'] in ['fullyconn']:
                self.layers.append(FullyConnectedLayer(layers[j]['n_neurons'],
                                                       self.layers[j-1].n_neurons,
                                                       layers[j]['activation']))
            else:
                raise NotImplementedError('only fully connected layers implemented at the moment')
        self.layers.append(OutputLayer(layers[-1]['n_neurons'],
                                       self.layers[-1].n_neurons,
                                        layers[-1]['activation'],
                                        layers[-1]['loss_type']))
        
    def forward(self,X):
        """
        Performs one forward sweep through the network.
        
        Parameters
        ----------
        X: numpy array
            design matrix, rows=samples, columns=features
        """
        self.layers[0].forward(X)
        for j in range(1,self.n_layers):
            self.layers[j].h = self.layers[j].forward(self.layers[j-1].h)
            
    def loss(self,y):
        J = self.layers[-1].compute_loss(y)
        return J
    
    def backward(self,y):
        g = self.layers[-1].gradient_loss(y)
        for j in range(1,self.n_layers):
            g = self.layers[-j].backward(g,self.layers[-j-1].h)
```

The forward and backward sweep are implemented by simply iterating through the layers in the correct order and calling the layer's backward and forward function implementation. They perform one step from the current layer to the next:

```python%start_inline=true
class InputLayer(Net):
    def __init__(self,batch_size,n_features):
        self.layer_type = 'input'
        self.n_inputs = batch_size
        self.n_neurons = n_features
    def forward(self,X):
        self.h = X
        return X
    def backward(self,g_prev_layer):
        pass
```

The input layer is simple. It just feeds the input to the next layer. 

```python%start_inline=true
class FullyConnectedLayer(object):
    def __init__(self,n_neurons,n_inputs,activation):
        self.layer_type = 'fullyconn'
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.activation = activation
        if self.activation not in ['relu','linear']:
            raise NotImplementedError('Activation not implemented')
        self.w = np.random.randn(self.n_inputs,self.n_neurons)*np.sqrt(1.0/self.n_inputs)#weights
        self.b = np.zeros([1,self.n_neurons])+0.1*(self.activation=='relu')#bias
        
    def forward(self,h_prev_layer):
        a = np.dot(h_prev_layer,self.w)+self.b
        self.a = a
        if self.activation == 'relu':
            h = np.maximum(a,0)
        elif self.activation =='linear':
            h = a
            
        self.h = h
        return h
    
    def backward(self,g,h_prev_layer): #next layer as in closer to the output layer
        if self.activation == 'relu':
            f_prime = ((self.a>0)*1.0)
        elif self.activation == 'linear':
            f_prime = np.ones_like(g)
            
        g = g*f_prime
        g_back = np.dot(g,self.w.T)
        #gradient descent
        self.b -= 0.1/10*np.sum(g,axis=0,keepdims=True)
        self.w -= 0.1/10*np.dot(h_prev_layer.T,g)
        
        
        return g_back
```

Our hidden layers do the following: They initializae the weights and biases and implement a farward and backward pass. The backward pass, for now, also performs the gradient update. The learning rate, unfortunately, is for now hard-coded into the definition of the hidden layer.

```python%start_inline=true
class OutputLayer(object):
    def __init__(self,n_neurons,n_inputs,activation,loss_type):
        self.layer_type = 'output'
        self.activation = activation
        if self.activation not in ['softmax','linear']:
            raise NotImplementedError('Activation not implemented')
        self.loss_type = loss_type
        if self.loss_type not in ['neg_log_likelihood','mse']:
            raise NotImplementedError('Loss function not implemented')
        self.n_neurons = n_neurons #=number of classes
        self.n_inputs = n_inputs
        self.w = np.random.randn(self.n_inputs,self.n_neurons)*np.sqrt(1.0/self.n_inputs)#weights
        self.b = np.zeros([1,self.n_neurons])#bias
        
    def forward(self,h_prev_layer):
        a = np.dot(h_prev_layer,self.w)+self.b
        self.a = a
        if self.activation == 'softmax':
            #print(a)
            a_max = np.max(a,axis=1,keepdims=True)
            #print(a_max)
            h = np.exp(a - a_max)
            h_sum = np.sum(h,axis=1,keepdims=True)
            h = h/h_sum
        elif self.activation == 'linear':
            h = a
        self.h = h
        return h
    
    def compute_loss(self,y):
        if self.loss_type=='neg_log_likelihood':
            return np.sum(-y*np.log(self.h),axis=1).mean()
        elif self.loss_type=='mse':
            return np.square(y-self.h).mean()
    
    def gradient_loss(self,y):
        if self.loss_type=='neg_log_likelihood':
            return np.mean(-y/self.h,axis=0,keepdims=True)
        elif self.loss_type=='mse':
            return 2/10*(self.h - y)
        
        
    def backward(self,g,h_prev_layer):
        if self.activation == 'softmax':
            f_prime = self.h * (1 - self.h)
        elif self.activation == 'linear':
            f_prime = 1.0
            
        g = g*f_prime
        
        g_back = np.dot(g,self.w.T)
        
        #gradient descent
        self.b -= 0.1/10*np.sum(g,axis=0,keepdims=True)
        self.w -= 0.1/10*np.dot(h_prev_layer.T,g)
                
        return g_back
```

The output layer looks similar in that it shares the same forward and backward functionionality. But, additionally, the output layer needs to compute the loss as well as the gradient of the loss to kick off the backpropagation algorithm. That's why the Neural Network class starts the backward loop by calling `gradient_loss` of the output layer.

Note, that in order to make this work for mini-batches, we only have to compute the averages 'mean()' along the correct axis.

To get an idea, if this is actually working as expected let's do a very simple experiment:

```python%start_inline=true
net = Net(layers=[{'layer_type':'input','batch_size':10,'n_features':5},
         {'layer_type':'fullyconn','n_neurons':5,'activation':'relu'},
         {'layer_type':'output','activation':'softmax',
          'loss_type':'neg_log_likelihood','n_neurons':2}],
    learning_rate=0.1)
X = np.array([[1,1,1,0,0],
              [1,2,1,0,-1],
              [2,1,1,-1,0],
              [2,2,2,0,0],
              [1,2,2,0,-1],
              [0,0,0,1,1],
              [-1,0,0,2,1],
              [0,-1,0,1,2],
              [-1,-1,0,2,2],
              [0,-1,0,1,1]])
X = (X-X.mean(0))/X.std(0)
y = np.array([[0,1],
              [0,1],
              [0,1],
              [0,1],
              [0,1],
              [1,0],
              [1,0],
              [1,0],
              [1,0],
              [1,0]])

net.forward(X)
losses = [net.loss(y)]
h1 = net.layers[-1].h
net.backward(y)
for j in range(99):
    net.forward(X)
    losses.append(net.loss(y))
    net.backward(y)
print(h1)
print()
print(net.loss(y))
print(net.layers[-1].h)
plt.figure()
plt.plot(losses)
```

This is a simple binary classification set up where we run the same batch through the network 100 times. 

A regression problem could look like this:
```python%start_inline=true
net = Net(layers=[{'layer_type':'input','batch_size':10,'n_features':5},
         {'layer_type':'fullyconn','n_neurons':5,'activation':'relu'},
         {'layer_type':'output','activation':'linear',
          'loss_type':'mse','n_neurons':1}],
    learning_rate=0.1)
X = np.array([[1,1,1,0,0],
              [1,2,1,0,-1],
              [2,1,1,-1,0],
              [2,2,2,0,0],
              [1,2,2,0,-1],
              [0,0,0,1,1],
              [-1,0,0,2,1],
              [0,-1,0,1,2],
              [-1,-1,0,2,2],
              [0,-1,0,1,1]])
X = (X-X.mean(0))/X.std(0)
y = np.array([[1],
              [2],
              [2],
              [4],
              [3],
              [0],
              [1],
              [1],
              [2],
              [0]])

net.forward(X)
losses = [net.loss(y)]
h1 = net.layers[-1].h
net.backward(y)
for j in range(99):
    net.forward(X)
    losses.append(net.loss(y))
    net.backward(y)
print(h1)
print()
print(net.loss(y))
print(net.layers[-1].h)
plt.figure()
plt.plot(losses)
```
Running both snippets, we see that usually we can see a continous decrease of our loss function after each iteration. 

### Next steps

So far, the parameters (like the learning rate and the initialization scheme of $$W$$ and $$b$$) are hand-coded into the definition of the classes. This is unsatsifactory. For example, we would like to specify the initialization scheme when we define our network architecture. However, the learning is not really part of our network architecture. It seems to be tied closer to the learning procedure. Therefore, it makes to sense to separate the learning algorithm from the network architecture. 

To not only deal with code reorganization, we will also introduce how to easily incorporate well-known regularization into our gradient descent update rules. The math is very basic, so the next part should be easy to follow. Saty tuned.

---
title: Wirtinger Gradients with Python
lesson: 02
layout: single
sidebar:
  nav:
    pulse_characterization
scholar:
  bibliography: phase_retrieval
---

# Adapting Scipy's minimize function to operate on Wirtinger gradients in the context of phase retrieval

Efficient optimization techniques are constructed 
by taking gradients and Hessians of cost functions 
to determine direction of descent or in trust-region
techniques for selecting minima of locally valid models.
Optimization is typically used with cost functions over
real vector variables, but this is not mandatory. 

$$\min_x f(x), x\in \mathbb{C}^N$$

Cost functions over complex variables do however 
have a slight complication that needs to be accounted for
if we want to select a complex vector that minimizes a 
cost function. A complex variable has two degrees of 
freedom per entry in $$x$$, and both of these need to 
be accounted for. While we are traditionally used to
accounting for this with a real and imaginary component,
Wirtinger calculus allows us to perform calculus 
directly on a complex variable and its conjugate.
Note I follow the standard notation of $$z$$ and 
$$z^* $$ for a complex variable and its conjugate.

This feature is particularly useful because many cost
functions over complex variables are naturally written
as a function of complex variables and their conjugate.
To steal an example from Adali et al {%cite impropriety%}
(a fantastic source on multiple topics in complex valued
math for signal processing), we examine the cost function
$$ f(z) = |z|^4 $$. This function can obviously be written
as the sum of a real and a complex 
variable $$ z = z_r + jz_i$$
whose terms can be expanded like so:

$$
\begin{aligned}
f(z) &= |z|^4 \\
    &= (z_r+ jz_i)^2 (z_r - jz_i)^2 \\
    &= z_r^4 + 2 z_r^2 z_i^2 + z_i^4 \\
\end{aligned}
$$

Differentiating with respect to $$z_r$$ and $$z_i$$ we 
get the following two expressions. Note we regroup back
into terms of $$z$$ and $$z^* $$.

$$
\begin{aligned}
\frac{\partial f}{\partial z_r} &= 4z_r^3 = 4z_i^2 z_r\\
&= 4z_r(z_r^2 + z_i^2) \\ 
&= 4z_r(z_r + j z_i)(z_r - j z_i) \\
&= 4z_r zz^* \\

\frac{\partial f}{\partial z_i} &= 4z_i^3 = 4z_r^2 z_i\\
&= 4z_i(z_r^2 + z_i^2) \\ 
&= 4z_i(z_r + j z_i)(z_r - j z_i) \\
&= 4z_i zz^* \\
\end{aligned}
$$

To see why these answers are significant, we turn to a 
slightly simpler operation; differentiation of our example
cost function with respect to the variable $$z$$ and 
$$z^* $$ while the other is held constant:

$$
\begin{aligned}
\frac{\partial f}{ \partial z} &= 2z z^* z^*  \\
\frac{\partial f}{ \partial z^* } &= 2z z z^*  \\
\end{aligned}
$$

We see that from this expression that we can actually 
write these two resulting pairs as linear combinations
of the other:

$$
\begin{align}
\frac{\partial f}{\partial z} &= 
    \frac{1}{2}\left( \frac{\partial f}{\partial z_r}
               -j \frac{\partial f}{\partial z_i}\right) \\
\frac{\partial f}{\partial z^* } &= 
    \frac{1}{2}\left( \frac{\partial f}{\partial z_r}
               +j \frac{\partial f}{\partial z_i}\right)
\end{align}
$$

What this means for us is that we get to work with a 
simpler set of variables while differentiating complex
cost functions while remaining functionally equivalent
to differentating with respect to real and imaginary
parts. In Wirtinger calculus, we differentiate a complex
function $$f(z, z^* )$$  with respect to both $$z$$ and 
$$z^* $$ holding the other constant. All the old rules 
of purely real calculus hold including chain rule and 
product rule. All your favorite functions will obey 
familiar differentiation rules. I like this fairly 
comprehensive tutorial: {% cite wirtinger_tutorial%}.

For use in optimization however, the cost function will be
real; we like to be able to order a cost function and 
say that one point is better or "lower" than another 
and this is poorly defined for complex numbers. In this 
case, the derivative of the cost function with respect
to $$z$$ and $$z^* $$ will merely be conjugates of each 
other. To perform optimization however, we would like
to take our result of differentation and generalize it 
to the notion of a gradient, called throughout the 
Wirtinger gradient. I was first introduced to this 
topic through papers on Wirtinger Flow in the problem 
of phase retrieval, which is the example I will give in 
Scipy here today. The original paper on Wirtinger descent
applied to phase retrieval, as well as a personal favorite
on studying it geometrically can be found here 
respectively: {%cite PR_Wirtinger%} {%cite geometry%}.

To form the gradient, we sequentially take the 
Wirtinger derivative of a cost function with respect
to each of its $N$ complex variables to form a complex 
row vector {%cite geometry%}. Note we will denote this 
as differentiating with respect to a complex vector, which 
we will signify with bold font like this: $$\mathbf{z}$$.
Concatenating the row vector derivatives against the 
variables $$\mathbf{z}, \mathbf{z}* $$, we arrive at the 
Wirtinger gradient $$\nabla f$$ by taking the complex
tranpose:

$$
\begin{align}
\frac{\partial f}{\partial \mathbf{z}} &= 
  \left( \frac{\partial f}{\partial z_0}, \dots, 
         \frac{\partial f}{\partial z_{N-1}} \right) \\
\frac{\partial f}{\partial \mathbf{z}^* } &= 
  \left( \frac{\partial f}{\partial z_0^* }, \dots, 
         \frac{\partial f}{\partial z_{N-1}^* } \right) \\
\nabla f &= \left( \frac{\partial f}{\partial \mathbf{z}},
        \frac{\partial f}{\partial \mathbf{z}^* } \right)^H
\end{align}
$$

We use $$f$$ alone as a short-hand but understand
that the function is dependent on both $$\mathbf{z}$$
and $$\mathbf{z}^*$$.

We summarize the relationship between the Wirtinger
gradient and the gradient with respect to real and 
imaginary componenets individually ($$\nabla_R f$$)
with the matrix
$$T_N$$ defined bellow {%cite impropriety%}:

$$
\begin{align}
T_N &= \begin{pmatrix}
    \mathbb{I}_N & j \mathbb{I}_N \\
    \mathbb{I}_N & -j \mathbb{I}_N 
\end{pmatrix} \\
\nabla f &= T_N \nabla_R f
\end{align}
$$

where $$I_N $$ is the identity matrix of dimension 
$$N\times N$$. We note that the Wirtinger Hessian 
can also be defined in a simililar manner 
{%cite geometry%}:

$$
\begin{align}
\nabla^2 f &= \begin{pmatrix}
\frac{\partial}{\partial \mathbf{z}}\left(\frac{\partial f}{\partial \mathbf{z}}\right)^H  &&
\frac{\partial}{\partial \mathbf{z}^*}\left(\frac{\partial f}{\partial \mathbf{z}}\right)^H \\
\frac{\partial}{\partial \mathbf{z}}\left(\frac{\partial f}{\partial \mathbf{z}^*}\right)^H  &&
\frac{\partial}{\partial \mathbf{z}^*}\left(\frac{\partial f}{\partial \mathbf{z}^*}\right)^H \\
\end{pmatrix}
\end{align}
$$

The Hessian with respect to the real and imaginary parts of
$$f$$ are realated to the Wirtinger Hessian through the
following tranform:

$$
\begin{align}
\nabla^2 f &= T_N \nabla_R^2 f T_N^H
\end{align}
$$

To go back and forth between the two representations
we'll need an inverse for $$T_N$$:

$$
\begin{align}
T_N^{-1} &= \frac{1}{2}T_N^H
\end{align}
$$

## Phase Retrieval

Phase retrieval is an awesomely simple "devil in the 
details" non-convex optimization problem that have 
had some very powerful approaches applied to it.
For an introduction to the problem and an overview of 
recent and historical solutions, 
I recommend perusing {%cite review%} or {%cite thesis%}.
We're going to use the intensity based definition similar
to the one in {%cite PR_Wirtinger%}:

$$
\begin{align}
\text{find } & \mathbf{z}\\
\text{such that }& y_i = |\mathbf{a}_i^H \mathbf{z}|^2 \\
i &\in \left[ 0,1,2,..., M-1 \right]
\end{align}
$$

Here, the measurements $$y_i$$ are intensity measurements
of the inner product between a known vector 
$$\mathbf{a}_i \in \mathbb{C}^N$$ 
and the unknown vector we'd like to 
recover $$\mathbf{z} \in \mathbb{C}^N$$. We note that
there can often be strictly non-negative noise added to 
each variable $$y_i$$. Often times a least-squares 
approach is taken to minimize the error between 
measurements and their reconstruction 
{%cite PR_Wirtinger geometry%}. We'll go ahead and
form the cost function for this least squares approach
here:

$$
\begin{align}
\min_\mathbf{z} f(\mathbf{z}, \mathbf{z}^*) &= 
\min_\mathbf{z} \frac{1}{2 M}\sum_{i=0}^{M-1} \left(
       y_i - |\mathbf{a}_i^H \mathbf{z} |^2\right)^2 \\
        & \downarrow \\
\min_\mathbf{z} f(\mathbf{z}, \mathbf{z}^*) &= 
       \min_\mathbf{z} \frac{1}{2 M}\sum_{i=0}^{M-1} \left(
        y_i - \mathbf{z}^H\mathbf{a}_i
        \mathbf{a}_i^H \mathbf{z}
       \right)^2
\end{align}
$$

This description is suggestive of course because this
entire formulation can be written in terms of a complex
vector and its gradient. Finally, we a function we can apply
complex gradient and Hessian to. We will do one final 
augmentation of our cost function 
$$f(\mathbf{z}, \mathbf{z}^*)$$ to make it a little
easier to adapt to code by vectorizing it. We will
use the shorthand $$\mathbf{y}$$ to represent the 
formation of a vector from the measurements 
$$y_i, i\in [0,1,2,...,M-1]$$. The matrix $$A$$ 
is similarly formed with $$a_i$$ as its collumns.
We will abuse $$\odot$$ for element-wise products,
and the square in the following (being internal to the 
sum) is element-wise as well. Sum over index is 
assumed:

$$
\begin{align}
f(\mathbf{z}, \mathbf{z}^*) & = \frac{1}{2 M}
\sum \left( (z^H A)\odot(A^H z) - y \right)^2
\end{align}
$$

Applying the Wirtinger gradient we arrive at results
equivalent to those in {%cite geometry%}:

$$
\begin{align}
\frac{\partial f}{\partial \mathbf{z}} &= 
\frac{1}{M} A (\mathbf{r} \odot \mathbf{g}) \\
\mathbf{g} &= A^H z \\
\mathbf{r} &= (z^H A)\odot(A^H z) - y \\
\nabla f &= \left( \frac{\partial f}{\partial \mathbf{z}},
        \left( \frac{\partial f}{\partial \mathbf{z} } \right)^* \right)^H
\end{align}
$$

TO BE CONTINUED. 


{% bibliography --cited %}



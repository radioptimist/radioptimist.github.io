---
title: Wirtinger Gradients with Python
lesson: 02
layout: single
sidebar:
  nav:
    pulse_characterization
scholar:
  bibliography: phase_retrieval
classes: wide
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
We will abuse $$\odot$$ for element-wise products
(or an element multiplying a whole row for a  product
between a vector and matrix),
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

Similarly, we arrive parts for the Wirtinger Hessian:

$$
\begin{align}
\frac{\partial}{\partial \mathbf{z}}\left(\frac{\partial f}{\partial \mathbf{z}}\right)^H &= 
\frac{1}{M} A (\mathbf{r}_+ \odot A^H) \\
\frac{\partial}{\partial \mathbf{z}^*}\left(\frac{\partial f}{\partial \mathbf{z}}\right)^H &= 
\frac{1}{M} A (\mathbf{g}^2 \odot A^T) \\
\mathbf{g} &= A^H z \\
\mathbf{r}_+ &= 2 (z^H A)\odot(A^H z) - y \\
\end{align}
$$

where the full Hessian is constructed as:

$$
\begin{align}
nw &=\frac{\partial}{\partial \mathbf{z}}\left(\frac{\partial f}{\partial \mathbf{z}}\right)^H  \\
ne &= \frac{\partial}{\partial \mathbf{z}^*}\left(\frac{\partial f}{\partial \mathbf{z}}\right)^H \\
\nabla^2 f &= \begin{pmatrix}
nw & ne \\
ne^* & nw^* 
\end{pmatrix}
\end{align}
$$

## Python
Now the fun part, we're going to adapt all this into 
Python to use the powerful minimize function that 
comes with Scipy. Scipy optimizes over a real 
variable, so we will develop our Wirtinger gradient 
and Hessian over a complex variable that is wrapped in 
a real conversion function that minimize will call.

We begin by creating functions to generate test vectors
for recovery and test matrices to generate phaseless 
measurements. For both of these we will be using 
complex Gaussian distributed random numbers to demonstrate
arbitrary varibles. We generate a test function to 
create an arbitrary $$\mathbf{x}$$, an arbitrary $$A$$,
and their resulting phaseless measurements
$$\mathbf{y} = |A^H \mathbf{x}|$$.

{%highlight python linenos=table%}
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec

def cgaus(rows,cols):
    val = np.random.randn(rows,cols) +\
          np.random.randn(rows,cols) * 1j
    val /= np.linalg.norm(val,axis=0)[None,:]
    return val

def meas(X,A):
    y = np.abs(A.conj().T @ X)
    return y

def test_meas():
    N = 16
    m = 100
    X = cgaus(N,1)
    A = cgaus(N,m)
    y = meas(X,A)
    plot_setup(X,A,y)

def plot_setup(X,A,y):
    fig = plt.figure(figsize=[10,5])
    gs = gridspec.GridSpec(2,3,figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3= fig.add_subplot(gs[0,1:])
    ax4 = fig.add_subplot(gs[1,1:])
    ax1.imshow(np.real(A),aspect = 'auto',interpolation='none')
    ax2.imshow(np.imag(A),aspect = 'auto',interpolation='none')
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel("N")
    ax2.set_ylabel("N")
    ax2.set_xlabel("M")
    ax3.plot(np.real(X),label='real')
    ax3.plot(np.imag(X),label='imag')
    ax3.set_xlabel("N")
    ax4.plot(y,label='$y=|A^Hx|$')
    ax4.plot(np.real(A.conj().T @ X).flatten(),label='$[ A^H x ]_R$',linestyle='--')
    ax4.plot(np.imag(A.conj().T @ X).flatten(),label='$[ A^H x ]_I$',linestyle='--')
    ax4.set_xlabel("M")
    ax4.legend(ncol=3,loc='lower left')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')
    ax1.set_title("$A$, real")
    ax2.set_title("$A$, imag")
    ax3.set_title("$x$, real and imaginary")
    ax4.set_title("Measurement $y$, missing phase")
    fig.tight_layout()
    plt.show()
{%endhighlight%}

An example of a plot created by this code is here:
{%include figure popup=True image_path="/assets/images/pulse_characterization/phase_retreival/example_dataset.svg" caption="Example dataset generated for phase retrieval" %}

Next up, we need functions to create cost function,
the Wirtinger gradient and Hessian functions, 
and functions to wrap them
so we can call them from Scipy minimize. We will pass
a concatenated vector (real/imag) from Scipy, so 
this wrapping function will do the conversion to 
complex variables and their conjugate. The wrapping
functions are prepended with "scipy". We also
demonstrate that the values are identical in a test
function when transformed by the $$T_N$$ matrices
defined earlier.

{%highlight python linenos=table%}
def wgrad(x,y,A):
    # Wirtinger gradient of the standard PR cost function
    forward = A.conj().T@x
    resid =  np.abs(forward)**2 - y**2
    prod = (A @ (resid * forward)) / A.shape[1]
    return np.vstack([prod,prod.conj()])

def whess(x,y,A):
    # Wirtinger Hessian of the standard PR cost function
    forward = A.conj().T@x
    resid_plus =  2 * np.abs(forward)**2 - y**2
    nw = A @ (resid_plus*A.conj().T)
    ne = A @ (forward ** 2 * A.T)
    top = np.hstack([nw,ne])
    bottom = np.hstack([ne,nw]).conj()
    return np.vstack([top,bottom]) / A.shape[1]

def scipy_wgrad(x,y,A):
    N = x.shape[0]//2
    X = x[:N] + 1j * x[N:]
    X = X.reshape(N,-1)
    grad = wgrad(X,y,A)
    z = grad[:N]
    zc = grad[N:]
    return np.real(np.vstack([z + zc,1j * (zc - z)])).flatten()/2

def scipy_whess(x,y,A):
    N = x.shape[0]//2
    X = x[:N] + 1j * x[N:]
    X = X.reshape(N,-1)
    hess = whess(X,y,A)
    a,b,c,d = hess[:N,:N],hess[:N,N:],hess[N:,:N],hess[N:,N:]
    H = np.zeros_like(hess)
    H[:N,:N] =  a + b + c + d
    H[:N,N:] =(-a + b - c + d) * -1j
    H[N:,:N] =( a + b - c - d) * -1j
    H[N:,N:] =  a - b - c + d
    return np.real(H) / 4

def test_wirt():
    N = 16
    m = 100
    X = cgaus(N,1)
    A = cgaus(N,m)
    y = meas(X,A)
    pert = X + cgaus(N,1) * 1e-0

    grad = wgrad(pert,y,A)
    hess = whess(pert,y,A)

    ripert = np.vstack([np.real(pert),np.imag(pert)]).flatten()
    sgrad = scipy_wgrad(ripert,y,A)
    shess = scipy_whess(ripert,y,A)

    Tx = np.vstack([np.hstack([np.eye(N), 1j * np.eye(N)]),\
                    np.hstack([np.eye(N),-1j*np.eye(N)])])

    grad_cx = (Tx @ sgrad).flatten()
    grad = grad.flatten()
    grad_ri = (Tx.T.conj() @ grad).flatten()/2
    print(np.linalg.norm( grad - grad_cx))
    print(np.linalg.norm( grad_ri - sgrad.flatten()))

    hess_cx = Tx @ shess @ Tx.T.conj()
    hess_ri = Tx.T.conj() @ hess @ Tx  / 4
    print(np.linalg.norm(hess - hess_cx))
    print(np.linalg.norm(shess - hess_ri))
{%endhighlight%}

These evaluate to machine precision. One final detail
is that any optimization that recovers $$x$$ will be 
off by a global phase, so we have a utility function 
for recovered vectors that aligns phase to ground-truth
for the purpose comparison. This is a cheap trick to 
determine phase difference between the recovered 
vectors, we demix one vector with another, and take the 
median phase offset.

{%highlight python linenos=table%}
def align(xout,X):
    phasing = np.median(np.angle(xout.conj() * X.flatten()))
    xout *= np.exp(1j * phasing)
    return xout
{%endhighlight%}

This final test function puts everything together:
creating a test set, adding noise at controlled SNR,
initialize a starting point for optimization randomly,
calls minimize with "Newton-CG" options, and then plots
results. I encourage testing with various methods, 
a favorite of mine is Newton-CG and L-BFGS-B. Both 
converge (as my old mentor Scott would say)
"like a bat out of hell".

{%highlight python linenos=table%}
def test_grad_descent():
    N = 32
    m = N * 10
    X = cgaus(N,1)
    A = cgaus(N,m)
    y = meas(X,A)
    snr = 60
    noise = np.random.randn(m)**2
    noise/=np.linalg.norm(noise)/np.linalg.norm(y)
    y += noise[:,None] * 10**(-snr/20)
    pert = cgaus(N,1) 
    print(np.linalg.norm(pert - X))

    init = np.vstack([np.real(pert), np.imag(pert)]).flatten()
    x = minimize(cost,init,args=(y,A),\
                 method='Newton-CG',jac=scipy_wgrad,\
                 hess = scipy_whess,
                 options=dict(disp=True),tol = 1e-16)
    print(x)
    
    xout = x.x[:N] + x.x[N:] * 1j
    xout = align(xout,X)

    plot_result(xout,X, y, A)

def plot_result(xout,X,y,A):
    error = np.linalg.norm(xout - X.flatten())
    y_rec = np.abs(A.conj().T @ xout).flatten()
    y = y.flatten()

    fig = plt.figure(figsize=[10,5])
    gs = gridspec.GridSpec(2,2,figure=fig)
    ax1 = fig.add_subplot(gs[:,:1])
    ax2 = fig.add_subplot(gs[0,1:])
    ax3 = fig.add_subplot(gs[1,1:])

    ax1.semilogy(y,label='$y$')
    ax1.semilogy(y_rec,label='$y_{rec}$')
    ax1.semilogy(np.abs(y - y_rec),label='$|y-y_{rec}|$')
    ax1.legend()

    ax2.plot(np.abs(X.flatten()),label='x')
    ax2.plot(np.abs(xout),label='x_{rec}')
    ax2.plot(np.abs(xout - X.flatten()),label='|x - x_{rec}|')
    ax2.legend()

    ax3.plot(np.angle(X.flatten() * xout.conj()))
    ax1.set_title(r"$y = |A^H x| + w[n]$ and $y_{rec} = |A^H x_{rec}|$")
    ax2.set_title(r"$|x|$ and recovered $|x_{rec}|$")
    ax3.set_title(r"$ \angle (x \odot x_{rec}^*)$")

    ax2.xaxis.set_visible(False)
    ax3.set_xlabel("N")

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax1.set_xlabel("M")
    plt.show()
{%endhighlight%}

The output plot is shown here. After phase alignment, 
the result is quite comparable even with random 
initialization and noise.
{%include figure popup=True image_path="/assets/images/pulse_characterization/phase_retreival/result.svg" caption="Example recovery at 60dB SNR" %}

The relationship between noise, measurement count, 
and problem dimension are explored extensively in the 
literature including {%cite review%}. 
Thank you for perusing and I hope that this
demonstration helps you if you ever need to use Python
for Wirtinger descent problems!

{% bibliography --cited %}



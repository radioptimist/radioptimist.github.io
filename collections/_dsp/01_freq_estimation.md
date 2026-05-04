---
title: Practical CRLB Frequency Estimation
lesson: 01
layout: single
sidebar:
  nav:
    dsp
scholar:
  bibliography: dsp
date:   2026-04-12 20:00:00 -0600
classes: wide
---

## Keeping it simple
Often times in signal processing, estimating the center frequency of a 
digital complex baseband signal is a crucial ingredient in topics
spaning engineering, physics, and math. There are a lot of beautiful 
approaches to measuring center frequency that deserve their own discussion 
another time. 

The write-up I'm presenting here is a list of cheap and easy favorites
that I use to estimate center frequency of a single clean complex baseband
tone. All of these techniques have a time and a place. Some of them are 
convenient and achieve expected results, some of them have a couple 
more steps and achieve the highest quality result: Cramér Rao Lower Bound (CRLB).

A phenominal source of my own cotinuing education and excellent base for 
estimation theory can be found at Professor Fowler's Estimation theory course
website {% cite estimation %}. Another resource I like

## Framing the problem
We assume we are looking at a baesband collected complex sinusoid of the 
following model which declares our measured data $$x[n]$$ and our unknown
model parameters $$A, f, \phi$$.

$$
\begin{align}
x[n] &= A \exp{j \left(2 \pi f n + \phi\right)} + w[n], n \in [0,1,2,...,N-1]\\
A & \in \mathbb{R} > 0 \\
f & \in [-1/2, 1/2] \\
\phi & \in [-\pi, \pi] \\
w[n] & \sim \mathcal{CN}\left(0, \sigma \right)
\end{align}
$$

We note that our estimates will have to extract information in the 
presence of complex gaussian noise. The strength of this noise 
will be defined by measured by signal to noise ratio $$A^2 / \sigma^2$$.
In these problems, unknown variables often depend on each other and can 
contribute in difficulty to determining any one of them. In this particular 
problem in my experience however, estimation of $$A,\phi$$ is very often 
trivial in comparison to determining $$f$$ and are basically determined
for free once $$f$$ has been solved for. With that, we examine a few 
approaches I like to use for solving for $$f$$. For convenience 
later on we will frequently roll $$2\pi f$$ into a single term $$\omega$$, 
the angular frequency. To differentiate, I will typically refer to 
$$f$$ as the cycle frequency, and $$\omega$$ as the angular frequency.
For all problems, we use the complex baseband assumptions that the 
units of time are simply samples, and the units of cyclic frequency are 
cycles per sample.

## Mean-Squared Error
Mean-squared error (MSE) is basically the variance of error, and will be our 
primary metric for comparing the quality of recovery. The lower the 
mean-squared error, the better we're doing. Consider an estimator 
operating over many experiments (independent parameters held equal)
that is designed to measure the value $$X_i$$ given some experiment
index $$i$$. The estimated value $$\hat{X_i}$$ a is produced during
each experiment and the quality of the estimator is measured with the MSE:

$$
\begin{align}
    MSE_{\hat{X}} &= \frac{1}{K}\sum_{n=0}^{K-1}\left( X_i - \hat{X_i} \right)^2 \\
    i & \in [0,1,...,K-1]
\end{align}
$$

This can also naturally extend into another useful metric, the Root-Mean-Squared Error (RMSE)
which is represents the standard deviation of error of an estimator. 
We will occasionally also use the Median-Squared Error as a useful metric when looking 
at phase transition plots where error suddenly jumps. The median can sometimes reveal boundaries
better than mean since over many trials the median is less swayed by outerliers.

## Numerical Test Parameters
Each of these estimators will have their MSE confirmed numerically 
accross 50 trials over a range of signal lengths $$N$$ and $$SNR$$. 
We will take the MSE over the 50 trials and return results as an 
image over $$N$$ and $$SNR$$. We will also typically provide a
comparison to some analytic function that predicts MSE in the 
form of a histogram of differences between numerical results and 
theoretical results (measured in dB).

## Basic estimation: FFT 
This one is really simple. Given a set of measurements $$x[n], n \in [0,1,...,N-1]$$
we take the FFT and look for the peak absolute value and map that to frequency.
In Python, the code that does this is very easy:

{%highlight python linenos=table%}
def simple_peak(data):
    F = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data))
    return freq[np.argmax(np.abs(F))]
{%endhighlight%}

We can actually work out what the MSE of this technique should be before we even get 
started. The value of cycle frequency $$f$$ is assumed to uniformly random:
$$\mathcal{U}\left(-.5,.5\right)$$. The FFT transforms $$N$$ samples of signal uniformly
spaced in time into $$N$$ samples of uniformly spaced cyclic frequency taking values 
$$\left[0, 1/N, 2/N,...,(N/2-1)/N,-(N/2)/N, (-N/2 + 1)/N,...,-2/N,-1/N \right]$$. Note
I am using the even FFT bin labelings. Since we are selecting the peak frequency from a finite
set of $$N$$ bins spaced at $$1/N$$, the error we face in estimating $$f$$ is a 
uniform distribution of $$\mathcal{U}[-.5/N,.5/N]$$ frequency bins. The variance of this 
distribution is $$\frac{1}{12 N^2}$$ and should be the MSE of the FFT center 
frequency estimation. The results of the numerical test are shown here:

{%include figure popup=True image_path="/assets/images/dsp/freq_estimation/basic_fft_precision.svg" caption="FFT MSE" %}

Note the color range of the plot may seem off, but is meant to scale and compare
to other estimators that will acheive higher precision. An important feature 
to notice about this plot is the section to the left of the blue line in the 
lefthand figure, a cliff where error is suddenly quite high compared to the other side
of the line. This section breaks assumptions about the peak value in the frequency domain. 
To the left side of this line, we should not expect frequency estimators built on FFT
to function because noise power in the FFT will begin to take values higher than
the concentrated peak signal value. We analyze this phenomina deeper in the next section.

## When do FFT methods succeed or fail?

## Cramér-Rao Lower Bound
This topic definitely deserves its own set of articles, but to have an idea 
of what it means to achieve a high-precision result, we will derive here the 
CRLB for the frequency estimation problem. For the purposes of this discussion
the CRLB represents the best that any unbiased estimator could hope to do on average.
An estimator may get lucky now and again and achieve an estimate closer to the 
``correct'' answer, but given noise and a function dependent on unknown parameters,
the CRLB is the lowest MSE we can hope to achieve for any estimator used to 
determine these parameters. 

{% bibliography --cited %}

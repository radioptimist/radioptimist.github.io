---
title: Practical CRLB Frequency Esimtation
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
model parameters $$A, \omega, \phi$$.

$$
\begin{align}
x[n] &= A \exp{j \omega n + \phi}, n \in [0,1,2,...,N-1]\\
A & \in \mathbb{R} > 0 \\
\omega & \in [-\pi/2, \pi/2] \\
\phi & \in [-\pi, \pi]
\end{align}
$$

In these problems, unknown variables often depend on each other and can 
contribute in difficulty to determining any one of them. In this particular 
problem in my experience however, estimation of $$A,\phi$$ is very often 
trivial in comparison to determining $$\omega$$ and are basically determined
for free once $$\omega$$ has been solved for. With that, we examine a few 
approaches I like to use for solving for $$\omega$$.

## Mean-Square Error

## Basic estimation: FFT 
This one is really simple. Given a set of measurements $$x[n], n \in [0,1,...,N-1]$$
we take the FFT and look for the peak absolute value and map that to frequency.
The rule of thumb I use (and we can see against the data) is that for detection like
this to work in Gaussian noise, the length $$N$$ needs to be more than $$32/SNR$$.
Upon detection, recovery accuracy for this method typically matches:

$$
\begin{align}
MSE_{FFT} &= \frac{\pi}{N^2} \\
\end{align}
$$

## Cramér-Rao Lower Bound
This topic definitely deserves its own set of articles, but to have an idea 
of what it means to achieve a high-precision result, we will derive here the 
CRLB for the frequency estimation problem. For the purposes of this discussion
the CRLB represents the best that any unbiased estimator could hope to do on average.
An estimator may get lucky now and again and achieve an estimate closer to the 
``correct'' answer, but given noise and a function dependent on unknown parameters,
the CRLB is the best we can hope to know these parameters on average. 



{% bibliography --cited %}

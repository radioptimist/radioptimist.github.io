---
title: Introduction
lesson: 01
layout: single
sidebar:
  nav: pulse_characterization
---
## A brief overview of the problem of pulse-characterization

My disertation centered on the use of optimization and
compressive sensing to solve for the complex time domain
profile of ultrafast optical pulses. The techniques were 
highly effective, and through these tutorials I aim to 
give a brief overview of the techniques presented in 
my thesis along with some example code.

For a more in depth description of techniques I of course
recommend reading my [disertation](https://mines.primo.exlibrisgroup.com/permalink/01COLSCHL_INST/1jb8klt/alma998214358002341). If you're not in 
the mood to read I recommend the Youtube recording of 
my defense bellow.
{% include video id="AW8EBM_j3B8" provider="youtube" %}


## Some framing to help me organize thoughts while I flesh this out
$$ e^{j \theta}$$

We can do some latex. 

Ultimately I think the layout I want to follow is:

Introduction, problem from the top and talk about why
this problem is difficult. talk about how everything gets
integrated and how theres this difficulty with abs values
Talk about a couple existing techniques and maybe have
some links in page.

Some theory on Wirtinger gradient stuff, some code samples
and plots about recovering forwards from complex functions
maybe a cubic and quadratic term. split into real/imag
then differentiate. or do Wirtinger and make it much easier
on yourself. Maybe talk about holomorphic?

Non-linear functions and tensors
small point to talk about linearizing any non-linear 
function using tensors

Hierarchy of problems in ultrafast optics. Show some of 
the pictures from the thesis here, just good page for 
showing other problems.

Compressed sensing and this problem. maybe a mention 
of new compressed sensing and computational imaging page.
A desire to do computational holography

Primary contributions theory talk about how
I studied variations of the problem that integrate 
in spectrum and how I integrate in time, and talk about 
the primary problem here being one of the fourth order
and overlapping fourth order.

Primary contributions results -- pictures of recovered 
pulses, an example of how to do this tutorial, new code.


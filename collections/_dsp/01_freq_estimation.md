---
title: Practical CRLB Frequency Esimtation
lesson: 01
layout: single
sidebar:
  nav:
    dsp
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

## Cramér-Rao Lower Bound
This topic definitely deserves its own set of articles, but to have an idea 
of what it means to achieve a high-precision result, we will derive here the 
CRLB for the frequency estimation problem.



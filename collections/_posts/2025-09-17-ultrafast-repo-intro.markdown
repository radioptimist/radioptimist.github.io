---
layout: single
title:  "A Github repo for ultrafast pulse characterization code"
date:   2025-09-17 20:00:00 -0600
categories: Ultrafast-Optics
---

I’ve collected the bare-bones base level code that I developed while working on my thesis and I’ve pushed it to [github]: (https://github.com/radioptimist/quadspec)

It’s a bit rough right now, but I will continue to separate out functions from test code and provide worked example problems on representative datasets. I’m excited to add to this repository, and I will be adding a collection dedicated to
walk-through tutorials [here]:(site.url/pulse_characterization).

The README of the project is more up to date than what I can provide here in a post, but I’d like to give a little discussion here as to the future of the project. Currently I have just a wad of code that I’ve accumulated over about 4 years crammed together into three files. The files have functions and test code strewn throughout and is not in a final product form. To move this to something more usable I think my current course of action is this:

Separate core functions from experimental, moving experimental functions into a folder on their own.
Separate test functions from core, moving these to a folder on their own.
Adding an examples folder, distinct from testing to provide code used in examples on this site.
For a TLDR 1000 foot view of what the code accomplishes, I encourage you to checkout the background presented in this video:

<iframe src="https://www.youtube.com/embed/AW8EBM_j3B8" onload="this.width=screen.width;this.height=screen.height;">
</iframe>

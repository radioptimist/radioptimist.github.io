---
layout: single
title:  "Building an antenna tool -- Introduction"
date:   2025-09-07 20:00:00 -0600
categories: Antennas
---
Throughout the course of my academic and professional work with antennas, I have made the same suite of tools for myself half a dozen times. I have whittled down a lot of the details of a tool, and I will write a series of posts on the creation of a light-weight Python tool for :

 - Defining and manipulating the frames of antennas, explaining how they are sampled and a convenient coordinate system for doing so
 - Expanding on a lightweight software model of an antenna
 - Creating algorithms for rotation, resampling, and combination of collections of antenna elements
 - Utilizing optimizers to inform decisions made about the design of an array
 - Flesh-out a collection of engineering tools for the convenient use of these tools
 - Providing a [Pyvista](https://pyvista.org) and [PyQtGraph](https://www.pyqtgraph.org) set of plotting functions

---
title: Project Outline
lesson: 01
layout: single
sidebar:
  nav:
    handyhermit
date:   2026-01-03 20:00:00 -0600
gallery1:
  - image_path: /assets/images/handyhermit/outline/apache.jpg
    url: /assets/images/handyhermit/outline/apache.jpg
    title: "Apache 3800 case"
  - image_path: /assets/images/handyhermit/outline/cluster.jpg
    url: /assets/images/handyhermit/outline/cluster.jpg
    title: "Old Pi3 Cluster"
  - image_path: /assets/images/handyhermit/outline/placement.jpg
    url: /assets/images/handyhermit/outline/placement.jpg
    title: "Boards and power supply"
classes: wide
---

Over the past ten years of recreationally collecting and playing with 
electronics, I have accumulated a large volume of finished projects that
sit on a shelf collecting dust. They were excellent demonstrations of 
concept, but very few of them serve purposes that put them front and 
center on a day to day basis. A glaring counter-example of this is 
electronics that I have repurposed as home automation sensors, but that's
an article for another day. What I've been wanting lately is a format 
for building ongoing projects that packs away easily and can travel well
with a consistent set of modular modifications that I can accumulate, print, 
and cut out. Oh, and I'd like it to be inexpensive.

Simultaneously, I'm not ashamed to say that I'm an avid patron of the 
toolstore Harbor Freight. Their tools catch a lot of flack for being 
mixed quality but several of their lines are outstanding. I'm a 
huge fan of the US General tool-chests, and hopefully someday I'll be
able to report on Badland winches. One lineup in particular that I'm 
excited about is Apache cases. As far as I'm concerned they
feel like an exact duplicate of Pelican cases in terms of quality but
a fraction of the price. 
One in particular has caught my eye for electronics: 
the [Apache 3800](https://www.harborfreight.com/3800-weatherproof-protective-case-large-black-63927.html).
Insane deal, especially on sale or with a coupon.

For Christmas this year my wife got me an 
[Octopus 1.1](https://biqu.equipment/products/bigtreetech-octopus-v1-1) board,
a motor control board with a ludicrous number of stepper drivers. This board
combined with the 3800 and remains of an old project of mine gave me 
the inspiration to begin construction on a small-hardcase sized 6DOF robotic arm.
I've been very interested in constructing a 6DOF arm since I was introduced to the 
board on 3D printing forums, as well as by this 
[project](https://hackaday.io/project/197770-manipylator/log/232565-manipilator-part-1-where-to-start) that 
sets out to construct a 6DOF arm using a Klipper setup with the Octopus.
The Pi3 I want to use to control this project is being harvested from 
a four node Pi3 cluster that I built almost 10 years ago to learn
OpenMPI. 

Some features I really want to play with for this box are:
- A 6DOF arm that folds into a case that can be powered by a single plug
- Hosting wireless control for the unit from the Pi3 as an access point
- A computer vision based feeback system (absolute angle encoders are a bit steep)
- A camera or radio/dish mounted as the tool

Some research I really want to dive into once I get a framed and calibrated
6DOF arm include:
- An amateur star-tracker for pose estimation and radio astronomy setup
- A photogrammetry platform for studying implicit neural representations for meshes

As I get further along in this project I'll post more articles and try to keep 
files and scripts posted up to github.

{%include gallery id="gallery1"%}

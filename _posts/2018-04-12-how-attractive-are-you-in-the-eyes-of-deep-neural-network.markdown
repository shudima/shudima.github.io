---
layout: post
title: How Attractive Are You in the Eyes of Deep Neural Network?
date: 2018-04-12
description: A couple of months ago, South China University published a paper and a dataset about “Facial Beauty Prediction”. You can ...
---
A couple of months ago, South China University published a paper and a dataset about “Facial Beauty Prediction”. You can find it here. The data set includes 5500 people that have a score of 1 through 5 of how attractive they are. Here are some examples from the paper:
here



<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*MEG1LZPHtp72xaKuBH-JPg.png">
</div>
There are also several famous people in the set. This Julia Robert’s photo got an average score of 3.78:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*TYO2-vJXBBX8jAblvXdKHA.png">
</div>
This photo of a famous Israeli model Bar Refaeli got a score of 3.7:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*NoAGOF-PE6_QoLoqCeMesg.png">
</div>
These may look like low scores, but a score of 3.7 means that bar is more attractive than ~80% of the people in the dataset.
Along with the dataset, the authors trained multiple models that trying to predict attractiveness of a person based on the picture of the face.
In this post, I want to reproduce their result and check how attractive am I.

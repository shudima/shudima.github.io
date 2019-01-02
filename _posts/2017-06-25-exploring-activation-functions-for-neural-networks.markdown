---
layout: post
title: Exploring Activation Functions for Neural Networks
date: 2017-06-25
description: In this post, I want to give more attention to activation functions we use in Neural Networks. For this, I’ll solve the  ...
---
In this post, I want to give more attention to activation functions we use in Neural Networks. For this, I’ll solve the MNIST problem using simple fully connected Neural Network with different activation functions.
MNIST
MNIST data is a set of ~70000 photos of handwritten digits, each photo is of size 28x28, and it’s black and white. That means that our input data shape is (70000,784) and our output (70000,10).
I will use a basic fully connected Neural Network with a single hidden layer. It looks something like this:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*jYhgQ4I_oFdxgDD-AOgV1w.png">
</div>
There’re 784 neurons in the input layer, one for each pixel in the photo, 512 neurons in the hidden layer, and 10 neurons in the output layer, one for each digit.
In keras, we can use different activation function for each layer. That means that in our case we have to decide what activation function we should be utilized in the hidden layer and the output layer, in this post, I will experiment only on the hidden layer but it should be relevant also to the final layer.
keras
There are many activation functions, I’ll go over only the basics: Sigmoid, Tanh and Relu.
First, let’s try to not to use any activation function at all. What do you think will happen? here’s the code (I’m skipping the data loading part, you can find the whole code in this notebook):
notebook
{% highlight python %}
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
{% endhighlight %}


As I said, 784 input, 512 in the hidden layer and 10 neurons in the output layer. Before training, we can look at the network architecture and parameters using model.summary and model.layers:
model.summary and model.layers
{% highlight python %}
Layers (input ==&gt; output)
--------------------------
dense_1 (None, 784) ==&gt; (None, 512)
dense_2 (None, 512) ==&gt; (None, 10)
{% endhighlight %}



{% highlight python %}
Summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
output (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
None
{% endhighlight %}












Ok now we’re sure about our network’s architecture, let’s train for 5 epochs:
{% highlight python %}
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 3s - loss: 0.3813 - acc: 0.8901 - val_loss: 0.2985 - val_acc: 0.9178
Epoch 2/5
60000/60000 [==============================] - 3s - loss: 0.3100 - acc: 0.9132 - val_loss: 0.2977 - val_acc: 0.9196
Epoch 3/5
60000/60000 [==============================] - 3s - loss: 0.2965 - acc: 0.9172 - val_loss: 0.2955 - val_acc: 0.9186
Epoch 4/5
60000/60000 [==============================] - 3s - loss: 0.2873 - acc: 0.9209 - val_loss: 0.2857 - val_acc: 0.9245
Epoch 5/5
60000/60000 [==============================] - 3s - loss: 0.2829 - acc: 0.9214 - val_loss: 0.2982 - val_acc: 0.9185
{% endhighlight %}










{% highlight python %}
Test loss:, 0.299
<strong class="markup--strong markup--pre-strong">Test accuracy: 0.918</strong>
{% endhighlight %}

Test accuracy: 0.918
We’re getting not so great results, 91.8% accuracy on MNIST dataset is pretty bad. Of course you can say that we need much more than 5 epochs, but let’s plot the losses:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*j5Y04bszChAUN6ftpQUuaQ.png">
</div>
You can see that the validation loss is not improving, and I can assure you that even after 100 epochs it won’t improve. We can try different techniques to prevent overfitting, or make our network bigger and smarter in order to learn better and improve, but lets just try using sigmoid activation function.
sigmoid
Sigmoid function looks like this:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*Noj6FwSMKHV9R_wSbnTKvA.png">
</div>
It squashes the output into a (0,1) interval and it’s non linear. Let’s use it in our network:
{% highlight python %}
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
{% endhighlight %}


You can see that the architecture is exactly the same, we changed only the activation function of the Dense layer. Let’s train again for 5 epochs:
Dense
{% highlight python %}
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 3s - loss: 0.4224 - acc: 0.8864 - val_loss: 0.2617 - val_acc: 0.9237
Epoch 2/5
60000/60000 [==============================] - 3s - loss: 0.2359 - acc: 0.9310 - val_loss: 0.1989 - val_acc: 0.9409
Epoch 3/5
60000/60000 [==============================] - 3s - loss: 0.1785 - acc: 0.9477 - val_loss: 0.1501 - val_acc: 0.9550
Epoch 4/5
60000/60000 [==============================] - 3s - loss: 0.1379 - acc: 0.9598 - val_loss: 0.1272 - val_acc: 0.9629
Epoch 5/5
60000/60000 [==============================] - 3s - loss: 0.1116 - acc: 0.9673 - val_loss: 0.1131 - val_acc: 0.9668

Test loss: 0.113
<strong class="markup--strong markup--pre-strong">Test accuracy: 0.967</strong>
{% endhighlight %}













Test accuracy: 0.967
That’s much better. In order to understand why, let’s recall how our neuron looks like:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*lqqZ5v486yJfnP_7MQatAA.png">
</div>
Where x is the input, w are the weights and b is the bias. You can see that this is just a linear combination of the input with the weights and the bias. Even after stacking many of those, we will still be able to represent it as a single linear equation. That means, that it’s similar to a network without hidden layers at all, and this is true for any number of hidden layers (!!). We’ll add some layers to our first network and see what happens. it looks like this:
x
w
b
{% highlight python %}
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
{% endhighlight %}

{% highlight python %}
for i in range(5):
    model.add(Dense(512))
{% endhighlight %}

{% highlight python %}
model.add(Dense(10, activation='softmax'))
{% endhighlight %}
Here’ how the network looks like:
{% highlight python %}
dense_1 (None, 784) ==&gt; (None, 512)
dense_2 (None, 512) ==&gt; (None, 512)
dense_3 (None, 512) ==&gt; (None, 512)
dense_4 (None, 512) ==&gt; (None, 512)
dense_5 (None, 512) ==&gt; (None, 512)
dense_6 (None, 512) ==&gt; (None, 10)

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)             (None, 512)               401920    
_________________________________________________________________
dense_2 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_3 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_4 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_5 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_16 (Dense)             (None, 10)                5130      
=================================================================
Total params: 1,720,330
Trainable params: 1,720,330
Non-trainable params: 0
_________________________________________________________________
None
{% endhighlight %}


























These are the results after training for 5 epochs:
{% highlight python %}
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 17s - loss: 1.3217 - acc: 0.7310 - val_loss: 0.7553 - val_acc: 0.7928
Epoch 2/5
60000/60000 [==============================] - 16s - loss: 0.5304 - acc: 0.8425 - val_loss: 0.4121 - val_acc: 0.8787
Epoch 3/5
60000/60000 [==============================] - 15s - loss: 0.4325 - acc: 0.8724 - val_loss: 0.3683 - val_acc: 0.9005
Epoch 4/5
60000/60000 [==============================] - 16s - loss: 0.3936 - acc: 0.8852 - val_loss: 0.3638 - val_acc: 0.8953
Epoch 5/5
60000/60000 [==============================] - 16s - loss: 0.3712 - acc: 0.8945 - val_loss: 0.4163 - val_acc: 0.8767

Test loss: 0.416
<strong class="markup--strong markup--pre-strong">Test accuracy: 0.877</strong>
{% endhighlight %}













Test accuracy: 0.877

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*8b_sQGBowcgxZWwAvsipbA.png">
</div>
This is very bad. We can see that the network is unable to learn want we want. This is because without non-linearity our network is just a linear classifier and not able to acquire nonlinear relationships.
On the other hand, sigmoid is a non linear function, and we can’t represent it as a linear combination of our input. That’s what brings non linearity to our network and ables it to learn non linear relationships. Let’s try train the 5 hidden layer network again, this time with sigmoid activations:
sigmoid
sigmoid
{% highlight python %}
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 16s - loss: 0.8012 - acc: 0.7228 - val_loss: 0.3798 - val_acc: 0.8949
Epoch 2/5
60000/60000 [==============================] - 15s - loss: 0.3078 - acc: 0.9131 - val_loss: 0.2642 - val_acc: 0.9264
Epoch 3/5
60000/60000 [==============================] - 15s - loss: 0.2031 - acc: 0.9419 - val_loss: 0.2095 - val_acc: 0.9408
Epoch 4/5
60000/60000 [==============================] - 15s - loss: 0.1545 - acc: 0.9544 - val_loss: 0.2434 - val_acc: 0.9282
Epoch 5/5
60000/60000 [==============================] - 15s - loss: 0.1236 - acc: 0.9633 - val_loss: 0.1504 - val_acc: 0.9548

Test loss: 0.15
<strong class="markup--strong markup--pre-strong">Test accuracy: 0.955</strong>
{% endhighlight %}













Test accuracy: 0.955

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*fDgApDPMHswjjE6iLG0ftg.png">
</div>
Again, that’s much better. We’re probably overfitting, but we got a significant boost in performance just by using activation function.

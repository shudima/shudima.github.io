---
layout: post
title: Word Embeddings using Deep Neural Network IMAGE Classifier
date: 2018-09-06
description: It is well known that word embeddings are very powerful in many NLP tasks. Most of the current implementations use what  ...
---

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/2000/1*4MfbrOpTozBXqv1aPG0S1g.png">
</div>
It is well known that word embeddings are very powerful in many NLP tasks. Most of the current implementations use what called “distributional hypothesis” that states that words in a similar context have similar meanings. Humans can tell if two words are similar not only by the context that may be used but also by looking at or imagining an object that this word might represent. Many times similar objects are related to words with similar meanings, like ‘car’ and ‘truck’, ‘house’ and ‘building’, ‘dog’ and ‘cat’.
In this post, I want to explore this idea and try to create word embeddings that will preserve the semantic meanings of the words, and I’ll do it by using a Neural Network that was trained on images.
I want to create word embeddings using the pertained ResNet model, which is a very popular image classifier that was trained on the ‘ImageNet’ data set. Then, I want to use the new embeddings to build a translation system. ‘Car’ in Spanish looks very similar (or even exactly the same) as ‘Car’ in English. I want to use this to translate words from any language to English.
ResNet
You can find all the code here
here
As stated in ImageNet website “ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns)”. It was trained on the ImageNet dataset which contains over 14 million images and 1000 classes. I’ll create word vectors using the predicted values of the last inner layer of the model and assume that words have similar meanings if their corresponding images are similar.
WordNet
Let’s look at a small example. We have 4 words: truck, wine, car, and a bottle.

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*XRAtHB1kD6kXp4X2yMnhFA.png">
</div>
As humans, we can tell that the words “truck” and “car” are more similar than “truck” and “bottle” or that “wine” and “bottle” are more similar than “wine” and “car”. And indeed the images of these words look similar. Those objects look similar in real life.
Here I’ll create embeddings only for noun words as their images probably describe the actual word better than images of words like “to”, “and”, “the” etc.
First, we need to find a big list of nouns. We’ll use NLTK for this:
{% highlight python %}
nouns = set()
for synset in list(wordnet.all_synsets('n')):

    n = synset.name().split('.')[0]

    if len(n) &gt; 2 and n.isalpha():
        nouns.add(n)
{% endhighlight %}






After some cleaning, we have ~39,000 noun words. Actually, not all of them are nouns, but we’ll stick with them anyway.
Next, we want to get an image for every word we have. We’ll use Google image search for that. There’s a great python library called google_images_download which will help us with that. We want to rescale the image so all the images will be with the same size. Then, We’ll use ResNet to create the word vector. I will use the last layer of the network, right before the final softmax:
google_images_download 
google_images_download
ResNet
softmax
{% highlight python %}
resnet = ResNet50(weights='imagenet', 
                  include_top=False, 
                  pooling='avg')
{% endhighlight %}


{% highlight python %}
embeddings = KeyedVectors(2048)
{% endhighlight %}
{% highlight python %}
for word in nouns:
  response = google_images_download.googleimagesdownload()
  path = response.download({'keywords': word, 'limit': 1})[word][0]
  img = cv2.imread(path)
  img = scipy.misc.imresize(img, 224.0 / img.shape[0])
  img = img.reshape((1,) + img.shape)
{% endhighlight %}





{% highlight python %}
  embeddings[word] = resnet.predict(img)[0]
{% endhighlight %}
Let's look at our embeddings:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*1eAKCGRHUZx5icofd9Cdcw.png">
</div>
Looks pretty good. We can see that word ‘car’ and ‘truck’ more similar than ‘car’ and ‘dog’
{% highlight python %}
embeddings.similarity('car', 'truck') # = 0.419
embeddings.similarity('car', 'dog') # = 0.270
{% endhighlight %}

This is not surprising, this is just the similarity between images of ‘car’, ‘truck’ and ‘dog’! We used the inner layers of the ResNet model to get a general representation of those images so it is easier for us to compare them.
ResNet
Our embeddings may preserve the semantic similarities between words, but it misses a very interesting part that other embeddings do have. In our embeddings, we can’t do something like queen — king = woman — man because our embeddings capture only the similarity between two words/objects, but it doesn't capture more complex relationships between words. We’re not using the English language here, we just look at similarities between images.
queen — king = woman — man
Now I want to build a simple translation system using my new embeddings. This is pretty straightforward, given any word in any language, we’ll use Google images search to find a corresponding image, then we’ll use ResNet to predict the final layer values, and finally, find the most similar English word to those predicted values:
ResNet
{% highlight python %}
def translate(word):
  response = google_images_download.googleimagesdownload()
  path = response.download({'keywords': word, 'limit': 1})[word][0]
  img = cv2.imread(path)
  img = scipy.misc.imresize(img, 224.0 / img.shape[0])
  img = img.reshape((1,) + img.shape)
{% endhighlight %}





{% highlight python %}
  vector = resnet.predict(img)[0]
{% endhighlight %}
{% highlight python %}
  return embedding.most_similar([vector])[0]
{% endhighlight %}
Lets see some examples:
{% highlight python %}
&gt;&gt;&gt; translate("מכונית") # 'car' in Hebrew
Output: clio # Brand of car (Renault Clio)
&gt;&gt;&gt; translate("ristorante") # 'restaurant' in Italian
Output: 'grubstake' # A restaurant in San Fransisco
&gt;&gt; Translate("еда") # 'meal' in Russian
Output: thanksgiving
{% endhighlight %}





As you can see it is not at all perfect but it does output some interesting translations.
This “embedding” is actually a simple “K Nearest Neighbors” model on top of ResNet representation of images. It’s not really translating words, but one nice thing about this is that we are able to compare (or classify) many thousands of type of words/images/classes while the original ResNet model was trained on 1000 classes only.
ResNet
ResNet
Final Note
Final Note
This is an interesting experiment (for me at least), but as you probably imagine to yourself, it is not so useful nor practical. Downloading ~39K photos takes a lot of time and there is no guarantee that the downloaded images are what we want or need. One of the most common tasks that may benefit from proper embeddings is text classification. I tried to use this embedding on the “20 Newsgroup” classification problem, but without success. If you’re interested in it, you can find the code here, it is a pretty messy notebook. It shows 3 experiments, random trainable embeddings, GLOVE non-trainable embeddings, and my visual embeddings. Enjoy!
here

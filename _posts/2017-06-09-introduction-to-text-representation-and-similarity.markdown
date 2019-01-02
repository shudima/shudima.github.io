---
layout: post
title: Introduction to text representation and similarity
date: 2017-06-09
description: The most basic thing in Machine Learning and Data Mining tasks is the ability to compare objects. We must compare (and s ...
---
The most basic thing in Machine Learning and Data Mining tasks is the ability to compare objects. We must compare (and sometimes average) objects in clustering, classification, query, and others. A text is no exception, and in this post, I want to explore different representations/embedding of texts and some of the most popular distance/similarity functions.
First, I want to talk about what properties we would like to have for text representations and distance/similarity functions:
Identical text must have the same representation and distance of zero (maximum similarity).When we have multiple texts, t1, t2, and t3, we want to have the ability to say that t1 is more similar to t2 than t3.Similarity/Distance should express the semantic comparison between texts, and text length should have little effect.
Identical text must have the same representation and distance of zero (maximum similarity).
When we have multiple texts, t1, t2, and t3, we want to have the ability to say that t1 is more similar to t2 than t3.
Similarity/Distance should express the semantic comparison between texts, and text length should have little effect.
So let’s start, consider these three sentences:
{% highlight python %}
s1 = "David loves dogs"
s2 = "Dogs are ok with David"
s3 = "Cats love rain"
{% endhighlight %}


Now let’s say we want to arrange these sentences into 2 groups. It is obvious that sentences 1 and 2 are in the same group. But how can we compare them programmatically?
For that, we’ll define 2 functions: 1) a Vectorizer function that gets some sentence and returns the vector that represents this sentence. And 2) a distance function that receives two vectors and returns how they are distant or similar.
The most basic way to represent a sentence is using a set of words. In most cases, we’ll want to pre-process our text to reduce noise. Pre-processing may include lowering case, stemming, removing stop words, and others. In our case, we’ll bring the words to its basic form and remove stop words. So the sentences are:
{% highlight python %}
s1 = ("david", "love", "dog")
s2 = ("dog","ok","david")
s3 = ("cat","love","rain")
{% endhighlight %}


To compare these sentences, we can use Jaccard Similarity. Jaccard Similarity is the proportion between a number of common words (Intersection) and a total number of words (union) of two sentences. The union of sentences 1 and 2 is (“David,” “love,” “dogs,” “ok”) and the intersection is (“David,” ”dog”), so Jaccard Similarity will be 2/4 = 0.5.
Jaccard Similarity
On the other hand, Jaccard Similarity between sentences 1 and 3 is 1/6 = 0.166, and the similarity between sentences 2 and 3 is 0/6 = 0.
Although this method solves our problem, there are some disadvantages. The best way to generalize these disadvantages is that both the representation and distance function are not “mathematical,” i.e. we can’t do sentence average, for example, manipulate the distance function using other mathematical functions (like taking a derivative).
To solve that, we have to represent our sentences in a more scientific way: vectors.
vectors
This is pretty simple: we’ll build a vocabulary with all the words in our corpus (all our sentences), and each word has an index. In our example, it looks like this:
{% highlight python %}
{'cat': 0, 'david': 1, 'dog': 2, 'love': 3, 'ok': 4, 'rain': 5}
{% endhighlight %}
We’ll represent each sentence as a 6-dimensional vector, as well as store 1 for every word in its index and zeros everywhere else. Here are our sentences:
{% highlight python %}
s1 = [0, 1, 1, 1, 0, 0]     
s2 = [0, 1, 1, 0, 1, 0]    
s3 = [1, 0, 0, 1, 0, 1]   
{% endhighlight %}


Now, each sentence is just a point (or a vector) in a 6-dimensional space. Here is how it looks like after we visualize it on 2-dimensional space using PCA:
Sentence 1 and 2 — Blue, Sentence 3 — Red
<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*aeayaMhGOMzPg1lc4wvjwg.png">
</div>
Sentence 1 and 2 — Blue, Sentence 3 — Red
You can see that the points of sentence 1 and 2 are closer than each one of them to sentence 3. There are many ways to express the distance between them. The most intuitive is “Euclidean distance” (also called L2 norm) which calculates this line:
Euclidean distance

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*kn4DxzjOMWG1ml1C-V66pQ.png">
</div>
Of course, there’s no need to implement it yourself; there are plenty implementations out there (numpy.linalg.norm in python for example).
So, here are the values of the Euclidian Distance:
{% highlight python %}
                 ╔══════╦════════╦════════╦════════╗
                 ║      ║   S1   ║   S2   ║   S3   ║  
                 ╠══════╬════════╬════════╬════════╣
                 ║ S1   ║    0   ║  1.41  ║    2   ║
                 ╠══════╬════════╬════════╬════════╣
                 ║ S2   ║  1.41  ║   0    ║  2.44  ║
                 ╠══════╬════════╬════════╬════════╣
                 ║ S2   ║    2   ║  2.44  ║    0   ║
                 ╚══════╩════════╩════════╩════════╝
{% endhighlight %}


It looks good, but still there’s some drawbacks. Many times, the text semantics are determined by the number of appearances of some word. For example, if we read an article about “Vectors,” then the term “Vector” will appear many times. In that case, using just binary values in our text vectors will omit the true semantics of the text. The simplest solution is just to store the number of appearances of the word in the right position in the vector.
Consider these sentences:
{% highlight python %}
s1 = "David loves dogs dogs dogs dogs"
s2 = "Dogs are ok with David"
s3 = "Cats love rain"
{% endhighlight %}


Of course, the first sentence is not grammatically correct, but we can say that it is pretty much about dogs. Let’s see the corresponding vectors:
{% highlight python %}
s1 = [0, 1, 5, 1, 0, 0]     
s2 = [0, 1, 1, 0, 1, 0]    
s3 = [1, 0, 0, 1, 0, 1]
{% endhighlight %}


The only difference between these vectors and the previous is that this time, we store 5 in the “dog” position of the first sentence vector.
Let’s plot those vectors using PCA again:
Sentence 1 — Blue, Sentence 2 — Green, Sentence 3 — Red
<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*rfdBr2EM572IC4YU8kWodQ.png">
</div>
Sentence 1 — Blue, Sentence 2 — Green, Sentence 3 — Red
See what happened; now, Sentence 2 (Green) and 3 (Red) are closer than 1 (Blue) and 2 (Green). This is very bad for us as we are not able to represent the semantics of the sentences.
The most typical solution to this problem is, instead of computing the distance between two points, we can compare the angle between two vectors, e.g., these values:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*m465Yl06qhuFxOjdxB90fg.png">
</div>
You can see that the angle of Sentence 1 and 2 is closer than 1 and 3 or 2 and 3. The actual similarity metric is called “Cosine Similarity”, which is the cosine of the angle between 2 vectors. The cosine of zero is 1 (most similar), and the cosine of 180 is zero (least similar).
Cosine Similarity
This way, we are able to represent the semantics of texts better, as well as compare text objects.
<h2>
Conclusion
</h2>
We saw the different ways to represent text and to compare text objects. There is no better or best way, there are many types of problems and challenges in text machine learning and data mining related task, and we should understand and pick the right option for us.
There are many other representations and distance/similarity function, including TfIdf and Word2Vec as very common text representations (of which I hope I’ll write another post).

---
layout: post
title: A (very) friendly introduction to Confidence Intervals
date: 2018-03-20
description: Today I want to talk about a basic term in statistics — confidence intervals, I want to do it in a very friendly manner, ...
---

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/2000/1*HsXmYjfHKoKOOzyCTPjkoA.jpeg">
</div>
Today I want to talk about a basic term in statistics — confidence intervals, I want to do it in a very friendly manner, discussing only the general idea, without too much fancy statistics terms and with python!
Although this term is very basic, it sometimes very hard to fully understand (as it was for me) what’s really going on, why we need it and when should we use it.
So let’s start.
Suppose you want to know what percentage of people in the U.S. who love soccer. The only one thing you can do in order to get a 100% correct answer to that question is by asking each one of the citizens in the U.S. whether or not they love soccer. According to Wikipedia, there’re over 325 million people in the U.S. It is not really practical to talk to 325 million people, so we got to think about something else, we have to get an answer by asking (much) less people.
We can do that by getting a random sample of people in the U.S. (talk to much less people) and get the percentage of people who love soccer in that sample, but then we won’t be 100% confident that this number is right or how far is this number from the real answer, so, what we’ll try to achieve is get an interval, for example, a possible answer to that question may be: “I am 95% confident that the percentage of people that love soccer in the U.S. is between 58% and 62%”. That’s where the name Confidence Interval come from, we have an interval, and we have some confidence about it.
Side note: Its very important that our sample will be random, we can’t just choose 1000 people from the city we live in, because then it won’t represent the whole U.S. population well. Another bad example, we can’t send Facebook messages to 1000 random people, because then we’ll get a representation of U.S. Facebook users, and of course not all of the U.S. citizens use Facebook.
So let’s say we have a random sample of 1000 of people form U.S. and we see that among those 1000 people 63% love soccer, what can we assume (infer) about the whole U.S. population?
In order to answer that, I want us to look at it in a different way. Suppose we know (theoretically) the exact percentage of people in the U.S., let's say it’s 65%, what is the chance, that by randomly picking 1000 people, only 63% of them will love soccer? lets use python to explore this!
{% highlight python %}
love_soccer_prop = 0.65  # Real percentage of people who love soccer
total_population = 325*10**6  # Total population in the U.S. (325M)
{% endhighlight %}

{% highlight python %}
num_people_love_soccer = int(total_population * love_soccer_prop)
{% endhighlight %}
{% highlight python %}
num_people_dont_love_soccer = int(total_population * (1 - love_soccer_prop))
{% endhighlight %}
{% highlight python %}
people_love_soccer = np.ones(num_of_people_who_love_soccer)
{% endhighlight %}
{% highlight python %}
people_dont_love_soccer = np.zeros(num_
people_dont_love_soccer)
{% endhighlight %}

{% highlight python %}
all_people = np.hstack([people_love_soccer, people_dont_love_soccer])
{% endhighlight %}
{% highlight python %}
print np.mean(all_people)
{% endhighlight %}
{% highlight python %}
# Output = 0.65000000000000002
{% endhighlight %}
In this code I created a numpy array with 325 million people, for each one of them I store one if he/she loves soccer and zero otherwise. We can get the percentage of ones in the array by calculating the mean of it, and indeed it is 65%.
Now, lets take few samples and see what percentage do we get:
{% highlight python %}
for i in range(10):
    sample = np.random.choice(all_people, size=1000)
    print 'Sample', i, ':', np.mean(sample)
{% endhighlight %}


{% highlight python %}
# Output:
Sample 0 : 0.641
Sample 1 : 0.647
Sample 2 : 0.661
Sample 3 : 0.642
Sample 4 : 0.652
Sample 5 : 0.647
Sample 6 : 0.671
Sample 7 : 0.629
Sample 8 : 0.648
Sample 9 : 0.627
{% endhighlight %}










You can see that we’re getting different values for each sample, but the intuition (and statistics theory) says that the average of large amount of samples should be very close to the real percentage. Let’s do that! lets take many samples and see what happens:
{% highlight python %}
values = []
for i in range(10000):
    sample = np.random.choice(all_people, size=1000)
    mean = np.mean(sample)
    values.append(mean)
{% endhighlight %}




{% highlight python %}
print np.mean(values)
{% endhighlight %}
{% highlight python %}
# Output = 0.64982259999999992
{% endhighlight %}
We created 10K samples, checked what is the percentage of people who love soccer in each sample, and then just averaged them, we got 64.98% which is very close to the real value 65%. Let’s plot all the values we got:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*Z7-LbiracyRL9EClaNHXgQ.png">
</div>
What you see here is an histogram of all the values we got in all the samples, a very nice property of this histogram is that it very similar to the normal distribution. As I said I don’t want to use too much statistics terms here, but let’s just say that if we do this process a very large number of times (infinite number of times) we will get an histogram that is very close to the normal distribution and we can know the parameters of this distribution. In more simple words, we’ll know the shape of this histogram, so we’ll be able to tell exactly how many samples can get any range of values.
Here’s an example, we’ll run this simulation ever more times (trying to reach infinity):

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*j74fVM-J7c4GSoPyexJJFg.png">
</div>
First of all, we can see that the center (the mean) of the histogram is near 65%, exactly as we expected, but we are able to say much more just by looking at the histogram, for example, we can say, that half of the samples are larger than 65%, or, we can say that roughly 25% are larger than 67%, or even, we can say that (roughly) only 2.5% of the samples are larger than 68%.
At this point, many people might ask two important questions, “How can I take infinite number of samples?” and “How does it helps me?”.
Let’s go back to our example, we took a sample of 1000 people and got 63%, we wanted to know, what is the chance that a random sample of 1000 people will have 63% soccer lovers. Using this histogram, we can say that there’s a chance of (roughly) 25% that we’ll get a value that is smaller or equal to 63%. We don’t actually need to do the infinite samples, the theory says us, that it is somewhat probable that if we choose randomly 1000 people, only 63% of them will love soccer.
Side note #2: Actually, in order to all that (find the chance of range of values), we need to know, or at least estimate, the standard deviation of the population. As I want to keep things simple, I’ll leave it for now.
Let’s go back to the reality and the real question, I don’t know the actual percentage of soccer lovers in the U.S. I just took a sample and got 63%, how does it help me?
So we don’t know the actual percentage of people who love soccer in the U.S. What we do know, that if we took infinite number of samples it will look like this:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*txu8-J2imhlqIPDqsL3gPA.png">
</div>
Here μ is the population mean (real percentage of soccer lovers in our example), and σ is the standard deviation of the population.
μ 
If we know this (and we know the standard deviation) we are able to say that ~64% of the samples will fall in the red area or, more than 95% of the samples will fall outside the green area in this plot:

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*GX0Qft3_Js0O9gbd9BeCNA.png">
</div>
If we use the plots before when we assumed that the actual percentage is 65%, than 95% of the samples will fall between 62% and 68% (+- 3)

<div class="img_row">
<img class="col three" src="https://cdn-images-1.medium.com/max/1600/1*HNJGj99HzbreV5eMJFt_OQ.png">
</div>
Of course the distance is symmetric, so if the sample percentage will fall 95% of the time between real percentage-3 and real percentage +3, then the real percentage will be 95% of the times between sample percentage -3 and sample percentage +3.
If we took a sample and got 63%, we can say that we 95% confident that the real percentage is between 60% (63 -3) and 66% (63+3).
This is the Confidence Interval, the interval is 63+-3 and the confidence is 95%.
I hope confidence intervals make more sense now, as I said before, this introduction misses some technical but important parts. There are plenty of articles that do contain these parts, and I hope that now it will be much easier to follow them.

# Distilling BERT — How to achieve BERT performance using Logistic Regression

BERT is awesome, and it’s everywhere. It looks like any NLP task can benefit from utilizing BERT. The authors [showed](https://arxiv.org/abs/1810.04805) that this is indeed the case, and from my experience, it works like magic. It’s easy to use, works on a small amount of data and supports many different languages. It seems like there’s no single reason not to use it everywhere. But actually, there is. Unfortunately, in practice, it is not so trivial. BERT is a huge model, more than 100 million parameters. Not only we need a GPU to fine tune it, but also in inference time, a CPU (or even many of them) is not enough. It means that if we really want to use BERT everywhere, we need to install a GPU everywhere. This is impractical in most cases. In 2015, this [paper](https://arxiv.org/abs/1503.02531) (by Hinton et al.,) introduced a way to distill the knowledge of a very big neural network into a much smaller one, like teacher and student. The method is very simple. We use the big neural network predictions to train the small one. The main idea is to use **raw** predictions, i.e, predictions before the final activation function (usually softmax or sigmoid). The assumption is that by using raw values, the model is able to learn inner representations better than by using “hard” predictions. Sotmax normalizes the values to 1 while keeping the maximum value high and decreases other values to something very close to zero. There’s little information in zeros, so by using raw predictions, we also learn from the not-predicted classes. The authors show good results in several tasks including MNIST and speech recognition.

Not so long ago, the authors of this [paper](https://arxiv.org/pdf/1903.12136.pdf) apply the same method to ... BERT! They show that we can get the same performance (or even better) on a specific task by distilling the information from BERT into a much smaller BiLSTM neural network. You can see their results in the table below. The best performance was achieved using BiLSTM-Soft, which means “soft predictions”, i.e, training on the raw logits and not the “hard” predictions. The datasets are: **SST-2** is Stanford Sentiment Treebank 2, **QQP** is Quora Question Pairs, **MNLI** isThe Multi-genre Natural Language Inference.

![](https://cdn-images-1.medium.com/max/2400/1*cgCdD3zXxlYfryryh-Y9mQ.png)

In this post, I want to distill BERT into a much simpler Logistic Regression model. Assuming you have a relatively small labeled dataset and a much bigger non-labeled dataset, the general framework for building the model is:

1.  Create some baseline on the labeled dataset
2.  Build a big model by fine-tuning BERT on the labeled set
3.  If you got good results (better than your baseline), calculate the raw logits for your unlabeled set using the big model
4.  Train a much smaller model (Logistic Regression) on the now pseudo-labeled set
5.  If you got good results, deploy the small model anywhere!

If you’re interested in a more basic tutorial on fine-tuning BERT, please checkout out my previous post:

[**BERT to the rescue!**
_A step-by-step tutorial on simple text classification using BERT_towardsdatascience.com](https://towardsdatascience.com/bert-to-the-rescue-17671379687f "https://towardsdatascience.com/bert-to-the-rescue-17671379687f")[](https://towardsdatascience.com/bert-to-the-rescue-17671379687f)

I want to solve the same task (IMDB Reviews Sentiment Classification) but with Logistic Regression. You can find all the code in [this](https://github.com/shudima/notebooks/blob/master/Distilling_Bert.ipynb) notebook.

As before, I’ll use `torchnlp` to load the data and the excellent [PyTorch-Pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) to build the model.

There are 25,000 reviews in the train set, we’ll use only 1000 as a labeled set and another 5,000 as an unlabeled set (I also choose only 1000 reviews from the test set to speed things up):

<pre name="0b12" id="0b12" class="graf graf--pre graf-after--p">train_data_full, test_data_full = imdb_dataset(train=True, test=True)
rn.shuffle(train_data_full)
rn.shuffle(test_data_full)
train_data = train_data_full[:1000]
test_data = test_data_full[:1000]</pre>

The first thing we do is create a baseline using logistic regression:

<iframe width="700" height="250" src="/media/02f0619bdc748f14b3f6238c04c9b9ab?postId=69a7fc14249d" data-media-id="02f0619bdc748f14b3f6238c04c9b9ab" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars2.githubusercontent.com%2Fu%2F7312293%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" allowfullscreen="" frameborder="0"></iframe>

We get not so great results:

<pre name="47b1" id="47b1" class="graf graf--pre graf-after--p">      precision    recall  f1-score   support</pre>

<pre name="4157" id="4157" class="graf graf--pre graf-after--pre">neg       0.80      0.80      0.80       522
pos       0.78      0.79      0.78       478</pre>

<pre name="139f" id="139f" class="graf graf--pre graf-after--pre">accuracy                      0.79      1000</pre>

Next step, is to fine-tune BERT, I will skip the code here, you can see it the [notebook](https://github.com/shudima/notebooks/blob/master/Distilling_Bert.ipynb) or a more detailed tutorial in my previous [post](https://towardsdatascience.com/bert-to-the-rescue-17671379687f). The result is a trained model called `BertBinaryClassifier` which uses BERT and then a linear layer to provide the pos/neg classification. The performance of this model is:

<pre name="253e" id="253e" class="graf graf--pre graf-after--p">       precision    recall  f1-score   support</pre>

<pre name="0e03" id="0e03" class="graf graf--pre graf-after--pre">neg       0.88      0.91      0.89       522
pos       0.89      0.86      0.88       478</pre>

<pre name="033e" id="033e" class="graf graf--pre graf-after--pre">accuracy                      0.89      1000</pre>

Much much better! As I said — Magic :)

Now to the interesting part, we use the unlabeled set and “label” it using our fine-tuned BERT model:

<iframe width="700" height="250" src="/media/3a06f3ae65d6cfe0cc60d0feb44195d3?postId=69a7fc14249d" data-media-id="3a06f3ae65d6cfe0cc60d0feb44195d3" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars2.githubusercontent.com%2Fu%2F7312293%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" allowfullscreen="" frameborder="0"></iframe>

We get:

<pre name="a844" id="a844" class="graf graf--pre graf-after--p">        precision    recall  f1-score   support</pre>

<pre name="8f36" id="8f36" class="graf graf--pre graf-after--pre">neg       0.87      0.89      0.88       522
pos       0.87      0.85      0.86       478</pre>

<pre name="b94d" id="b94d" class="graf graf--pre graf-after--pre">accuracy                      0.87      1000</pre>

Not as great as the original fine-tuned BERT, but it’s much better than the baseline! Now we are ready to deploy this small model to production and enjoy both good quality and inference speed.

Here’s another reason to [5 Reasons “Logistic Regression” should be the first thing you learn when becoming a Data Scientist](https://towardsdatascience.com/5-reasons-logistic-regression-should-be-the-first-thing-you-learn-when-become-a-data-scientist-fcaae46605c4) :)

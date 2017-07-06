+++
date = "2017-07-05T00:05:01-04:00"
title = "tSNE"
tags = ["Dimensionality Reduction", "Python"]
categories = [ "Machine Learning" ]
project_url = "https://github.com/ndeepesh/MachineLearning-Experiments/blob/master/experiments/tSNE.ipynb"
+++

Recently, I came across an interesting non-linear visualization or dimensionality reduction method - tSNE(Distributed Stochastic Neighbour Embedding). [Here](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) is the original paper describing the method in much more detail.

tSNE is just like PCA where you can reduce the dimensionality of data to 2 or 3 dimensions to visualize it. But unlike PCA, tSNE does that using non-linear ways. It converts everything to probability distributions. I am listing down the steps on a high level for a bigger picture. <br/>

1. Assume we have points of shape=(n, m). We'll call this original or higher dimension<br/>
2. We need to convert these points to shape=(n, 2). We'll call this lower or embedded dimension.<br/>
3. Calculate Pairwise similarity matrix for all data points(example eculedian distance between all pairs)<br/>
4. Each point is then converted to a probability distribution using the above similarity matrix<br/>
5. Now, we initialize n points using a normal distribution but in the lower dimension.<br/>
6. Again we convert these new points to a probability distribution<br/>
7. Finally, run gradient-descent on the two distributions(point 4 and 6) with kl-divergence between them as the cost function <br/>

[Here](https://github.com/ndeepesh/MachineLearning-Experiments/blob/master/experiments/tSNE.ipynb) is the link to the Jupyter Notebook. I had to dig into sklearn source code a bit to understand the algorithm completely. For this I have extensive comments for anyone to understand it. Hope it helps :) <br/>
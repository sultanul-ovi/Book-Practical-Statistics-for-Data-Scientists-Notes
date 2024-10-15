# Chapter 9 - Unsupervised Learning

Most applications of Machine Learning today use Supervised Learning. If you remember, this essentially means that you are providing the algorithm with both a set of inputs and outputs to train on (your X and y datasets). In unsupervised tasks, you will ask the ML algorithm to find correlations in the data without outputs to train on (just your X). Supervised learning depends on data being labeled to be executed. Because a majority of the data in the world is unlabeled, there is huge opportunity to use unsupervised learning to drive value. 

Some of the most common uses of Unsupervised Learning include:

- Clustering
    - This is where we attempt to group data based on the similarity of different dimensions
- Anomaly Detection
    - This tries to determine what "normal" data is and then point out the outliers which are abnormal
- Density Estimation
    - This process tries to estimate the Probability Density Function (PDF) of a random process, usually so that you can find anomalies which will be located in areas of low-density

# Clustering

Clustering is similar to classification where each member of a dataset is assigned to a group based on its features but different in that the groups aren't known prior to training. Clustering looks for similarities between datasets and will group values into one of many classes. You can usually define the number of classes that the algorithm can break data into. 

There are many uses of clustering from customer segmentation to reverse image search, one that's particularly interesting is semi-supervised learning. This is when some of the data is labeled but not all of it. You can use a clustering algorithm to help label the rest of the data. 

## K-Means

K-Means is an efficient and quick clustering algorithm that can be used to rapidly cluster relatively sparse datasets. Essentially, K-Means tries to find the "center" of as many groups of data as you specify and use those centers to cluster the data, with the cluster being defined by proximity to one of the n-points that are plotted. 

<aside>
💡

It's important to scale values prior to using K-Means as you might get "stretched" results if you don't

</aside>

```python
from sklearn.cluster import KMeans

k=5 # The number of clusters that you want to find
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X) # Assigns each instance to a group from 0-4

kmeans.cluster_centers_ # Reveals the centroids of each cluster

kmeans.predict(X_test) # Assigns each test value to a cluster
```

![Screen Shot 2021-08-25 at 7.15.28 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-25_at_7.15.28_PM.png)

As you can see, we've matched most records to what would be the obvious cluster. If you look between the pink and yellow boundaries though, you'll see there are a couple of records that aren't as easy to match as others. Because we used *hard-clustering* we simply matched each cluster to its closest center. If we want to get a score for each classification then we'd want to use *soft-clustering*. The score can simply be the distance from the centroid, it can also be a similarity measure like a Gaussian Radial Basis Function. You can use the `transform` method to calculate distance from a centroid. 

The K-Means algorithm randomly places the centroids in different parts of the space you're modeling in and then moves them until it finds the mathematically optimal solution. The graphic below shows one way this works: 

![Screen Shot 2021-08-25 at 7.22.47 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-25_at_7.22.47_PM.png)

It is possible for the centroid to converge at a suboptimal solution. In order to prevent this, the initialization of points is very important. 

### Voronoi Diagrams

Before getting to the initialization tricks we can use with K-Means, you might have noticed the interesting diagrams we've been looking at earlier. You can visualize the individual clusters using something called a Voronoi diagram. This diagram will visually show you boundaries of each cluster within two dimensions. The solution used was taken from this StackOverflow question: 

[Drawing boundary lines based on kmeans cluster centres](https://stackoverflow.com/questions/49347738/drawing-boundary-lines-based-on-kmeans-cluster-centres)

[Notebook on nbviewer](https://nbviewer.jupyter.org/gist/pv/8037100)

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

np.random.seed(1234)
points = np.random.rand(15, 2)

X = [i[0] for i in points]
y = [i[1] for i in points]
plt.scatter(X, y)
```

![Screen Shot 2021-08-25 at 7.10.51 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-25_at_7.10.51_PM.png)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)

# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.4)

plt.plot(points[:,0], points[:,1], 'ko')
plt.axis('equal')
plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

vor = Voronoi(points)
```

### Centroid Initialization Methods

**Manual Initialization**

If you now approximately where the centroids should be, you can specify these points using the `init` parameter when initializing you KMeans algorithm. 

```python
kmeans = KMeans(n_clusters=5, init=LIST_OF_POINTS, n_init=1)
```

An alternative method is to run the algorithm multiple times and keep the centroids from the best run. The `n_init` parameter is the parameter that specifies how many times the algorithm with run. 10 is the default amount. SKlearn uses the *inertia* value as a performance metric. It;s the mean squared distance between each instance and its closest centroid. 

**KMeans++**

KMeans++ is an improvement to the original KMeans algorithm proposed in 2006 that tends to pick centroids that are far away from one another. This is the default method used by SKLearn. You can set the `init` hyperparameter to "`random`" if you want to use the original, less-optimal method. 

### Increasing K-Means Speed

**Minibatches**

If you have data that is too large to fit in memory or works very slowly with your KMeans clustering algorithm you can use the `MiniBatchKMeans` algorithm to batch your data and send in those batches one-by-one. 

```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
```

This will typically lead to worse inertia than the regular K-Means algorithm. 

### Choosing the Optimal Number of Clusters

Choosing the optimal number of clusters is extremely important in a K-Means clustering algorithm. You can't just use inertia because inertia will go down as you add more clusters even if it doesn't make sense to. If you think about it, as you add more clusters the distance from each individual centroid will reduce as well. As the graph below demonstrates though, the returns start diminishing quite quickly.

![Screen Shot 2021-08-26 at 1.01.44 AM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-26_at_1.01.44_AM.png)

Although this method is rather simple, it is also very rough. A better but more computationally intensive method is to use the *silhouette score*. This is the mean *silhouette coefficient* over all instances. You can calculate the silhouette coefficient for an instance using $(b-a)/max(a,b)$. A is the mean distance to the other instances in the same cluster, and b is the mean nearest-cluster distance, i.e. the mean distance to the elements of the nearest cluster. The coefficient can be between -1 and 1, 1 meaning that the instance is deep inside its own cluster, and -1 meaning that the instance might have been assigned to the wrong cluster.  

```python
from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)
```

![Screen Shot 2021-08-26 at 1.07.17 AM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-26_at_1.07.17_AM.png)

You can also plot a *silhouette diagram* which is a plot of every silhouette coefficient sorted by the cluster it's a part of. The height represents the number of instances in that cluster, and the width is the sorted silhouette coefficients of the instances in the cluster (wider being better). The dashed line is the silhouette score. You want most of your clusters to fall to the right of this line. 

![Screen Shot 2021-08-26 at 1.11.19 AM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-26_at_1.11.19_AM.png)

In this instance we might want 5 clusters because of the relatively even heights and because all of the clusters area greater than the silhouette score. 

K-Means is not a great solution when clusters have varying sizes or different densities or non spherical shapes. The below chart illustrates such a non spherical dataset: 

![Screen Shot 2021-08-28 at 6.46.08 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-28_at_6.46.08_PM.png)

## Clustering for Image Segmentation

Clustering can be very useful for image recognition. When you're trying to recognize an image, oftentimes you're trying to pick out sections of the image in order to recognize objects like a pedestrian or a specific person. Segmenting out objects like this is called *semantic segmentation* all pixels that are part of the same object are grouped together. For an example, if there were multiple pedestrians in a picture, they would all be put together in the same segment. In *Instance Segmentation* all pixels that are a part of an individual object are segmented together. *Color segmentation* simply segments all pixels of the same pixel color together. If you were performing a task like determining the total area of forest in a satellite picture, this would be more than sufficient. 

The below code + the `ladybug.png` image found [here](https://github.com/ageron/handson-ml2/blob/master/images/unsupervised_learning/ladybug.png), can help demonstrate how color segmentation works. 

```python
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from PIL import Image as im

image = imread("ladybug.png")
image.shape # Height, Width, Color Channels

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

plt.show()
```

![Screen Shot 2021-08-28 at 8.05.17 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-28_at_8.05.17_PM.png)

You'll notice that the lady bug, which is bright red in color, is quickly absorbed into the other colors. Remember, KMeans prefers groups that are close to the same size. 

### Using Clustering for Preprocessing

Clustering can be an efficient technique for dimensionality reduction. An example would be using it to reduce the complexity of an image like the one that we looked at in the previous section. 

For an example lets use the digits dataset which is a MNIST-like dataset of greyscale images that represent the digits 0-9. 

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test) # Our baseline score

pipeline = Pipeline([ # Building a pipeline which we will use to improve our score through KMeans clustering first
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression())
])

pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test) # Get the score

param_grid = dict(kmeans__n_clusters=range(2, 100)) # Grid search for the best parameters

grid_clf = GridSearchCV(pipeline, param_grid, cv=3)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_ # Output the best number of clusters
```

## Clustering for Semi-Supervised Learning

One particularly useful application of clustering, is labeling datasets that are otherwise not labeled. 

First, we train our training data on a clustering algorithm creating a total of 50 different clusters that will act as representations for our digits. Note how we don't introduce the labels at all. This process doesn't require us to have labeled data

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import numpy as np

X_digits, y_digits = load_digits(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]
```

Then let's plot the data and "manually" label it. (Because I have the labels I'm just going to pull the labels from the dataset but 50 items is not too difficult to manually label).

```python
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

plt.show()
```

![Screen Shot 2021-08-29 at 3.56.11 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-08-29_at_3.56.11_PM.png)

```python
y_representative_digits = y_train[representative_digit_idx]
y_representative_digits # "Manually" labeling the above instances
```

Now let's fit the above data to a logistic regression model:

```python
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
```

And that's how you can classify data without actually labeling massive datasets yourself. 

You can even propagate the labels in these clusters to all instances within that cluster. You'll probably want to only include the instances in the X_train set which are closest to the centroids so that you don't get instances on the borders which the clustering algorithm is not sure belong to a certain class. 

```python
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

percentile_closest = 75

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

log_reg.score(X_test, y_test)
```

## DBSCAN

DBSCAN is a very different way of handling clustering from K-Means. 

- For each instance, the algorithm counts how many instances are distance $\epsilon$ (epsilon) from it. This is called the instances $\epsilon-neighborhood$.
- If an instance has at least `min_samples` instances in its neighborhood including itself, it's called a *core instance*. Basically, core instances are those in densely packed neighborhoods
- Multiple core instances can be strung together if close enough to be a single cluster
- Any instance that is not a core instance and doesn't have one in its neighborhood is called an anomaly

The implementation is very simple:

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

dbscan.labels_ # Labels of the clusters per instance. A "-1" indicates that the instance is an anomaly
```

# Other Clustering Algorithms

# Gaussian Mixtures

*Gaussian  mixture models (GMMs)* are probabilistic models that assumes instances were generated from a mixture of several Gaussian distributions with unknown parameters. They are a form of soft-clustering and the distributions typically look like ellipsoids. The book's explanation of the GMM is a bit confusing and I found one that made more sense to me over here: 

[Gaussian Mixture Models](https://www.youtube.com/watch?v=q71Niz856KE)

![Screen Shot 2021-09-02 at 4.17.05 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-09-02_at_4.17.05_PM.png)

We want to find the gaussian distribution that will lift the points the highest. 

![Screen Shot 2021-09-02 at 4.24.07 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-09-02_at_4.24.07_PM.png)

We do this by finding the center of mass of the points, then taking the variance of all of the dimensions to find the covariance. 

![Screen Shot 2021-09-02 at 4.29.29 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-09-02_at_4.29.29_PM.png)

First we start with random Gaussians as shown above and classify every point as a proportion of one of the two Gaussians

![Screen Shot 2021-09-02 at 4.29.59 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-09-02_at_4.29.59_PM.png)

We then fit our Gaussians based on the proportions of the colored points. We keep doing this until our algorithm converges (stops changing significantly).

When working with GMMs we use $\mu$ to represent the mean or the center of the Gaussian distribution, $\Sigma$ to represent the size, shape and orientation, and $\phi$ to represent the relative weights. 

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
```

Like a KNN algorithm, Gaussian distributions can converge on poor solutions which is why we need to run it several times and keep the best solution. 

We can check the weights, means, covariances, whether the model converged or not, and the number of iterations it took to converge using:

```python
gm.weights_

gm.means_

gm.covariances_

gm.converged_

gm.n_iter_
```

You can then predict what output a new instance will result in using:

```python
gm.predict(X)
gm.predict_proba(X)
```

Because it's a generative model, you can sample a few instances from it:

```python
X_new, y_new = gm.sample(6)
X_new
```

And then you can get the density of the model at any given location:

```python
gm.score_sample(X)
```

Convergence can be difficult to achieve when working with a high-dimensionality problem so we then would use the hyperparameter `covariance_type` and pick between :

- `spherical` - All clusters need to be spherical but can have different radii (variances)
- `diag` - All custers need to take on an ellipsoidal shape
- `tied` - All custers must have the same ellipsoidal shape, size, and orientation (they must all have the same covariance matrix)
- `full` - The default parameter which says that a cluster can take on any shape, size, orientation

### Anomaly Detection Using Gaussian Mixtures

Any instance that lies in a low-density region can be considered an outlier with the user specifying the threshold that defines a low density region. 

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```

*Novelty detection* is a similar task that assumes that the dataset is clean and without outliers which is different from anomaly detection. 

### Selecting the Number of Clusters

With K-means you'll use the silhouette score or inertia to select the appropriate number of clusters. You can't use these with non-spherical Gaussian distributions as they're not reliable. You want to instead use a number of clusters that reduce *theoretical information criterion* such has the *Bayesian information criterion* (BIC) or the *Akaike information criterion* (AIC). 

AIC and BIC tend to pick the same model and prioritize models that have fewer parameters. 

```python
gm.bic(X)
gm.aic(X)
```

![Screen Shot 2021-09-02 at 10.14.58 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-09-02_at_10.14.58_PM.png)

### Bayesian Gaussian Mixture Models

As long as you provide a number of clusters that's greater than what you expect the problem to require, the `BayesianGaussianMixture` class will give you weights close to 0 for unnecessary clusters,

```python
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(X)
np.round(bgm.weights_,2)
```

![Screen Shot 2021-09-02 at 10.25.18 PM.png](Chapter%209%20-%20Unsupervised%20Learning%2012067492d83f80deb0a8dd31c8d30eb3/Screen_Shot_2021-09-02_at_10.25.18_PM.png)

As you can see from the output, the algorithm was able to figure that only three clusters were necessary. 

# Other Algorithms for Anomaly and Novelty Detection

PCA

Fast-MCD (Minimum covariance determinant)

Isolation Forest

Local Outlier Factor (LOF)

One-class SVM

# Extra Content

[10 Clustering Algorithms With Python - Machine Learning Mastery](https://machinelearningmastery.com/clustering-algorithms-with-python/)

[A Guide to Data Clustering Methods in Python](https://builtin.com/data-science/data-clustering-python)
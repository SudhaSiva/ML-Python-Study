# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:15:13 2017

@author: Sudhakar

Unsupervised Learning packages -must required
"""
# Kmeans clustering algorithm points
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3)
model.fit()
model.predict()
centroids = model.cluster_centers_

# Evaluating a cluster
model.inertia_ #  inertia  should be mimimum for better clustering

# the below methods is same as using mode.fit and model.predict seperately
labels = model.fit_predict(samples)
# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

from sklearn.preprocessing import StandardScaler #  Scaling the features to vary between o and 1
from sklearn.preprocessing import Normalizer
# setting a pipeline
from sklearn.pipeline import make_pipeline 



# Heirarical clustering for better visualization -Dendrogram plot package
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.cluster.hierarchy import fcluster

#t-sne
from sklearn.manifold import TSNE

# PCA
from sklearn.decomposition import PCA
models.components_  # to get the PCA components)
pca.explained_variance_  (# helps to get bar chart of variance distribution)

from scipy.stats import pearsonr


from sklearn.decomposition import TruncatedSVD

# transforms list of documents to word frequency array
from sklearn.feature_extraction.text import TfidfVectorizer


# NMF dimensinality reduction technique

from sklearn.decomposition import NMF


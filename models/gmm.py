
from models.base import BaseAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import logging

#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class GMMAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.gm = GaussianMixture(n_components=2, random_state=0)
        self.data = data.X


    def fit(self):
        logging.info("scoring")
        self.gm.fit(self.data)

        #  https://github.com/vlavorini/ClusterCardinality/blob/master/Cluster%20Cardinality.ipynb
        labels= self.gm.predict(self.data)
        '''
        # https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
        The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
        '''
        score = metrics.silhouette_score(self.data, labels, metric='euclidean')
        logging.info(f'Silhouette Score is {score}')

        score = metrics.calinski_harabasz_score(self.data, labels)
        # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster
        logging.info(f'CH Score is {score}')

        # Zero is the lowest possible score. Values closer to zero indicate a better partition.
        score = metrics.davies_bouldin_score(self.data, labels)
        logging.info(f'Davies Bouldin Score is : {score}')


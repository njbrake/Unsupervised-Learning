
from models.base import BaseAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import logging

#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class KMeansAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4, random_state=0)
        self.data = data.X


    def fit(self):
        logging.info("scoring")
        self.kmeans.fit(self.data)
        labels = self.kmeans.labels_
        '''
        # https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
        The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
        '''

        n_clusters=np.arange(2, 20)
        sils=[]
        chs=[]
        dbs=[]
        for n in n_clusters:
            km = KMeans(init="k-means++", n_clusters=n, n_init=4, random_state=0).fit(self.data)
            labels=km.labels_
            score =metrics.silhouette_score(self.data, labels, metric='euclidean')
            sils.append(score)
        
            logging.debug(f'Silhouette Score is {score}')

            score = metrics.calinski_harabasz_score(self.data, labels)
            chs.append(score)
            # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster
            logging.debug(f'CH Score is {score}')

            # Zero is the lowest possible score. Values closer to zero indicate a better partition.
            score = metrics.davies_bouldin_score(self.data, labels)
            dbs.append(score)
            logging.debug(f'Davies Bouldin Score is : {score}')
        plt.plot(n_clusters, sils)
        plt.title("Silhouette Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.show()
        plt.clf()

        plt.plot(n_clusters, chs)
        plt.title("Calinski Harabasz Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.show()
        plt.clf()

        plt.plot(n_clusters, dbs)
        plt.title("Davies Bouldin Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.show()
        plt.clf()

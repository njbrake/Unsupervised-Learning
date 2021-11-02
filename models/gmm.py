
from models.base import BaseAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import logging

#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class GMMAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.name = 'GMM'
        self.k = 0

    def fit(self):
        if self.k == 0:
            raise Exception("find bes k first")
        
        gm = GaussianMixture(n_components=self.k)
        gm.fit(self.data.X)
        labels= gm.predict(self.data.X)
        return labels

    def find_best_k(self):
    
        #  https://github.com/vlavorini/ClusterCardinality/blob/master/Cluster%20Cardinality.ipynb
        n_clusters=np.arange(2, 20)
        bics=[]
        for n in n_clusters:
            gm = GaussianMixture(n_components=n)
            gm.fit(self.data.X)

            bics.append(gm.bic(self.data.X))
    
        plt.plot(n_clusters, bics)
        plt.title("BIC Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.savefig(f'out/gmm_{self.data.name}_bic.png', bbox_inches='tight')
        plt.clf()


        min = np.argmin(bics)
        if self.data.name == 'heart':
            best_k = 9
        else:
            best_k = 7
        self.k = best_k

        return best_k

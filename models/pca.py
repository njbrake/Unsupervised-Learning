
from models.base import BaseAlgorithm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
import logging



#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class PCAAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.name = 'PCA'


    def find_best_dimension(self):
        X = self.data.X
        y = self.data.y


        pca = decomposition.PCA(n_components=13)
        pca.fit(X)

        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title("Cumulative Explained Variance Ratios", fontsize=20)
        plt.xticks(np.arange(1,13))
        plt.xlabel("N. of Dimensions")
        plt.ylabel("Variance Ratio")
        plt.savefig(f'out/pca_{self.data.name}_variance.png', bbox_inches='tight')
        plt.clf()

        fig = plt.figure(1, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

        plt.cla()
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)


        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 3, 0,2]).astype(float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

        ax.w_xaxis.set_ticklabels([])
        ax.w_xaxis.set_label("Principal Component 1")
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

        plt.savefig(f'out/pca_scatter_{self.data.name}.png', bbox_inches='tight')

        plt.clf()

        self.k = 4


    
    def reduce(self):
        
        self.find_best_dimension()
        print("K is ",self.k)

        pca = decomposition.PCA(n_components=self.k)
        pca.set_params(n_components=self.k)
        val = pca.fit_transform(self.data.X)
        return val
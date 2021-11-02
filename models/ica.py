
from models.base import BaseAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
import logging



#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class ICAAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.k = 0
        self.name = 'ICA'


    def find_best_dimension(self):
        X = self.data.X
        y = self.data.y

        ica = decomposition.FastICA()
        k = []
        dims = np.arange(2,13)
        for dim in dims:
            ica.set_params(n_components=dim)
            val = ica.fit_transform(X)
            val = pd.DataFrame(val)
            val = val.kurt(axis=0)
            k.append(val.abs().mean())
        # Maximize kurtosis

        # see what features are maximizing independence
        
        plt.plot(dims, k)
        plt.title("Kurtosis", fontsize=20)
        plt.xticks(dims)
        plt.xlabel("N. of Dimensions")
        plt.ylabel("Kurtosis")
        plt.savefig(f'out/ica_{self.data.name}_kurt.png', bbox_inches='tight')
        plt.clf()

        fig = plt.figure(1, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

        plt.cla()
        ica = decomposition.FastICA(n_components=3)
        ica.fit(X)
        X = ica.transform(X)


        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 3, 0,2]).astype(float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

        plt.savefig(f'out/ica_scatter_{self.data.name}.png', bbox_inches='tight')
        plt.clf()

        self.k = 11

    def reduce(self):
        
        self.find_best_dimension()
        print("K is ",self.k)

        ica = decomposition.FastICA()
        ica.set_params(n_components=self.k)
        val = ica.fit_transform(self.data.X)
        return val

from models.base import BaseAlgorithm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.random_projection import GaussianRandomProjection
from sklearn import datasets
import logging
    




#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class RPAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.name = 'RP'


    def find_best_dimension(self):
        plt.clf()
        X = self.data.X
        dims = np.arange(2,13)
        for _ in range(50):
            reconstruction_correlation = []
            for dim in dims:
                # https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
                rp = GaussianRandomProjection(n_components=dim)
                distance_1 = pairwise_distances(rp.fit_transform(X))
                distance_2 = pairwise_distances(X)
                error = np.corrcoef(distance_1.ravel(),distance_2.ravel())[0,1]
                reconstruction_correlation.append(error)
            plt.plot(dims, reconstruction_correlation)
        plt.title("Reconstruction Correlation", fontsize=20)
        plt.xticks(dims)
        plt.xlabel("N. of Dimensions")
        plt.ylabel("Reconstruction Correlation")
        plt.savefig(f'out/rp_{self.data.name}_correlation.png', bbox_inches='tight')
        plt.clf()

        self.k = 9

    def reduce(self):
        
        self.find_best_dimension()
        print("K is ",self.k)

        gr = GaussianRandomProjection(n_components=self.k)
        gr.set_params(n_components=self.k)
        val = gr.fit_transform(self.data.X)
        return val
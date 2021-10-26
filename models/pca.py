
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
        self.data = data.X
        self.labels = data.y


    def fit(self):
        logging.info("scoring")
        X = self.data

        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)

        principle_df = pd.DataFrame(data = X, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Principal Component - 1',fontsize=20)
        ax.set_ylabel('Principal Component - 2',fontsize=20)
        ax.set_zlabel('Principal Component - 2',fontsize=20)
        plt.title("Principal Component Analysis",fontsize=20)

        used = set()
        targets = [x for x in self.labels if x not in used and (used.add(x) or True)]
        colors = ['r', 'g','b','y']
        for target, color in zip(targets,colors):
            logging.info(target)
            indicesToKeep = self.labels == target
            ax.scatter(principle_df.loc[indicesToKeep, 'principal component 1']
                    , principle_df.loc[indicesToKeep, 'principal component 2']
                    , principle_df.loc[indicesToKeep, 'principal component 3']
                    , c = color, s = 50)

        plt.legend(targets,prop={'size': 15})
        plt.show()
        plt.clf()
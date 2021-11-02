
from models.base import BaseAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class DTAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.name = 'DT'

    def find_best_dimension(self):
        X = self.data.X
        y = self.data.y


        sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
        sel.fit(X, y)
        sel.get_support()
        selected_feat= X.columns[(sel.get_support())]
        # print(selected_feat)
        # print(sel.get_support())
        # print(sel.estimator_.feature_importances_)
        
        num_bins = X.shape[1]
        xs = np.arange(num_bins)
        # the histogram of the data
        plt.scatter(xs, sel.estimator_.feature_importances_)
        for i in xs:
            plt.annotate(X.columns[i], (xs[i],sel.estimator_.feature_importances_[i]))

        plt.savefig(f'out/dt_{self.data.name}_importance.png', bbox_inches='tight')
        plt.clf()

        self.selected_feat = selected_feat
    
    def reduce(self):
        
        self.find_best_dimension()

        val = self.data.X[self.selected_feat]
        return val
        
import logging
import numpy as np

class DataClass():
    def __init__(self, name,X,y):
        self.name = name
        self.X = X
        self._y = y

    @property
    def y(self):
        """I'm the 'x' property."""
        # raise Exception('Youre not allowed to access y')
        return self._y

    def __str__(self) -> str:
        (n_samples, n_features), n_digits = self.X.shape, np.unique(self._y).size
        return f"Dataset: {self.name} \n# Labels: {n_digits}; # samples: {n_samples}; # features {n_features}"
import argparse
import json
import logging
import sys
import random as rand
import numpy as np
from datetime import datetime
import pandas as pd

from data.base import DataClass
from models.km import KMeansAlgorithm
from models.gmm import GMMAlgorithm
from models.pca import PCAAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s — %(name)s — %(levelname)s — %(filename)s:%(lineno)d — %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Learning!')
    # parser.add_argument('--clustering', action='store_true',
    #                     help='If true, run clustering')
    args = parser.parse_args()

    #  https://www.kaggle.com/ronitf/heart-disease-uci
    df = pd.read_csv("data/heart.csv")
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    heart_data = DataClass('heart', X,y)

    df = pd.read_csv("data/phone_price.csv")
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    mobile_data = DataClass('mobile', X, y)

    data = [heart_data, mobile_data]

    for da in data:
        # km = KMeansAlgorithm(da)
        logger.info(da)
        # km.fit()

        pca = PCAAlgorithm(da)
        pca.fit()




    results = []


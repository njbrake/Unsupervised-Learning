import argparse
import json
import logging
import sys, time
import random as rand
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from data.base import DataClass
from models.km import KMeansAlgorithm
from models.gmm import GMMAlgorithm
from models.pca import PCAAlgorithm
from models.ica import ICAAlgorithm
from models.rp import RPAlgorithm
from models.dt import DTAlgorithm

from sklearn.neural_network import MLPClassifier

from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s — %(name)s — %(levelname)s — %(filename)s:%(lineno)d — %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Learning!')
    # parser.add_argument('--clustering', action='store_true',
    #                     help='If true, run clustering')
    parser.add_argument('--part', help='The part of the assignment you\'re doing')
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

    logger.info(f'Running Part {args.part}')

    if args.part == 'all':
        stages = [1,2,3,4,5]
    else:
        stages = [int(args.part)]
    
    for stage in stages:
        for da in data:
            logger.info(da)
            
            if stage == 1:
                logger.info("Clustering Experiment")
                
                logger.info('K Means')
                km = KMeansAlgorithm(da)
                best = km.find_best_k()
                logger.info(f'Best k is {best}')

                logger.info("GMM")
                gmm = GMMAlgorithm(da)
                best = gmm.find_best_k()
                logger.info(f'Best k is {best}')
            elif stage == 2:
                logger.info("Dimensionality Reduction")

                logger.info("PCA")
                pca = PCAAlgorithm(da)
                pca.find_best_dimension()

                logger.info("ICA")
                ica = ICAAlgorithm(da)
                ica.find_best_dimension()

                logger.info("RP")
                rp = RPAlgorithm(da)
                rp.find_best_dimension()

                logger.info("Decision Tree")
                dt = DTAlgorithm(da)
                dt.find_best_dimension()

            elif stage == 3:

                # first do dimensionality reduction, then cluster
                dim_reductions = [PCAAlgorithm(da), ICAAlgorithm(da),RPAlgorithm(da),DTAlgorithm(da)]
                for reducer in dim_reductions:
                    red_da = DataClass(f'{da.name}_{reducer.name}_gmmreduced', reducer.reduce(), None)
                    red_da_km= DataClass(f'{da.name}_{reducer.name}_kmreduced', reducer.reduce(), None)
                    clusterers = [KMeansAlgorithm(red_da_km), GMMAlgorithm(red_da)]
                    for clust in clusterers:
                        best = clust.find_best_k()
                        logger.info(f"{da.name} {clust.name} {reducer.name} Best k is {best}")
        if stage == 4:
            '''
            Apply the dimensionality reduction algorithms to one of your datasets from assignment #1
                (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've
                already done this) and rerun your neural network learner on the newly projected data.
            '''
            #  Uncomment this to generate the learning curves for the baseline model
            # for nnlen in [2,3,5,10,25,50,75,100,200,400]:
            #     nn_layer_size=(nnlen,)
            #     da = data[1]
            #     X_train, X_test, y_train, y_test = train_test_split(da.X,da.y,test_size=0.2)
            #     clf = MLPClassifier(max_iter=16000, hidden_layer_sizes=nn_layer_size, learning_rate_init=0.001)
            #     # Create the learning curve visualizer
            #     visualizer = LearningCurve(
            #         clf,
            #         scoring='accuracy'
            #     )
            #     visualizer.fit(X_train, y_train)
            #     visualizer.finalize()            
            #     # Get access to the axes object and modify labels
            #     plt.savefig(f'out/nn_{da.name}_{nnlen}_baseline_curve.png', bbox_inches='tight')
            #     plt.clf()
            #     start = time.time()
            #     clf = MLPClassifier(max_iter=16000, hidden_layer_sizes=nn_layer_size, learning_rate_init=0.001)
            #     clf.fit(X_train,y_train)
            #     end = time.time()
            #     print("Training NN took" , (end - start)*1000, " ms")
            # first do dimensionality reduction, then cluster
            dim_reductions = [PCAAlgorithm(da), ICAAlgorithm(da),RPAlgorithm(da),DTAlgorithm(da)]

            for reducer in dim_reductions:
                red_da = DataClass(f'{da.name}_{reducer.name}_reduced', reducer.reduce(), da.y)

                X_train, X_test, y_train, y_test = train_test_split(red_da.X,red_da.y,test_size=0.2)
                for nnlen in [2,3,5,10,25,50,75,100,200,400]:
                    nn_layer_size=(nnlen,)
                    clf = MLPClassifier(max_iter=16000, hidden_layer_sizes=nn_layer_size, learning_rate_init=0.001)
                    # Create the learning curve visualizer
                    visualizer = LearningCurve(
                        clf,
                        scoring='accuracy'
                    )
                    visualizer.fit(X_train, y_train)
                    visualizer.finalize()            
                    # Get access to the axes object and modify labels
                    plt.savefig(f'out/nn_{red_da.name}_{nnlen}_curve.png', bbox_inches='tight')
                    plt.clf()
                    start = time.time()
                    clf = MLPClassifier(max_iter=16000, hidden_layer_sizes=nn_layer_size, learning_rate_init=0.001)
                    clf.fit(X_train,y_train)
                    end = time.time()
                    print("Training NN for ", reducer.name ,"took" , (end - start)*1000, " ms")
            
        if stage == 5:
            '''
            Apply the clustering algorithms to the same dataset to which you just applied the dimensionality
                reduction algorithms (you've probably already done this), treating the clusters as
                if they were new features. In other words, treat the clustering algorithms as if they 
                were dimensionality reduction algorithms. Again, rerun your neural network learner 
                on the newly projected data.
            '''
            for nnlen in [2,3,5,10,25,50,75,100,200,400]: 
                nn_layer_size=(nnlen,)
                da = data[1]
                dim_reductions = [PCAAlgorithm(da), ICAAlgorithm(da),RPAlgorithm(da),DTAlgorithm(da)]

                for reducer in dim_reductions:
                    # First reduce data
                    red_da = DataClass(f'{da.name}_{reducer.name}_reduced', reducer.reduce(), da.y)

                    # Now use clustering to reduce it even further
                    clusterers = [KMeansAlgorithm(red_da), GMMAlgorithm(red_da)]
                    for clust in clusterers:
                        
                        # cluster and use those as the new X data
                        clust.find_best_k()
                        A = clust.fit()
                        X = np.reshape(A, (A.shape[0],-1 ))
                        X_train, X_test, y_train, y_test = train_test_split(X,da.y,test_size=0.2)
                        clf = MLPClassifier(max_iter=16000, hidden_layer_sizes=nn_layer_size, learning_rate_init=0.001)
                        # Create the learning curve visualizer
                        visualizer = LearningCurve(
                            clf,
                            scoring='accuracy'
                        )
                        visualizer.fit(X_train, y_train)
                        visualizer.finalize()            
                        # Get access to the axes object and modify labels
                        plt.savefig(f'out/nn_clustered_{clust.name}_{reducer.name}_{red_da.name}_{nnlen}_curve.png', bbox_inches='tight')
                        plt.clf()
                        start = time.time()
                        clf = MLPClassifier(max_iter=16000, hidden_layer_sizes=nn_layer_size, learning_rate_init=0.001)
                        clf.fit(X_train,y_train)
                        end = time.time()
                        print("Training NN for ", reducer.name , clust.name, "took" , (end - start)*1000, " ms")


    results = []


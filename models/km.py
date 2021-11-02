
from models.base import BaseAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn import metrics
import logging

#  Inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
class KMeansAlgorithm(BaseAlgorithm):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.name = 'KM'
        self.k = 0

    def fit(self):
        if self.k == 0:
            raise Exception('find best k first')
        km = KMeans(init="k-means++", n_clusters=self.k, n_init=4)
        km.fit(self.data.X)
        labels=km.labels_
        return labels


    def find_best_k(self):
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
            km = KMeans(init="k-means++", n_clusters=n, n_init=4)
            km.fit(self.data.X)
            labels=km.labels_
            score =metrics.silhouette_score(self.data.X, labels, metric='euclidean')
            sils.append(score)
        
            logging.debug(f'Silhouette Score is {score}')

            # score = metrics.calinski_harabasz_score(self.data.X, labels)
            # chs.append(score)
            # # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster
            # logging.debug(f'CH Score is {score}')

            # # Zero is the lowest possible score. Values closer to zero indicate a better partition.
            # score = metrics.davies_bouldin_score(self.data.X, labels)
            # dbs.append(score)
            # logging.debug(f'Davies Bouldin Score is : {score}')

                # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(self.data.X, labels)

            y_lower = 10
            for i in range(n):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n)
                plt.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples


                # The vertical line for average silhouette score of all the values
                plt.axvline(x=score, color="red", linestyle="--")

            plt.xlabel("Silhouette Score")
            plt.ylabel("Sample #")
            plt.savefig(f'out/km_sil_{self.data.name}_{n}.png', bbox_inches='tight')
            plt.clf()
        plt.plot(n_clusters, sils)
        plt.title("Silhouette Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.savefig(f'out/km_{self.data.name}_silhouette.png', bbox_inches='tight')
        plt.clf()

        # plt.plot(n_clusters, chs)
        # plt.title("Calinski Harabasz Scores", fontsize=20)
        # plt.xticks(n_clusters)
        # plt.xlabel("N. of clusters")
        # plt.ylabel("Score")
        # plt.savefig(f'out/km_{self.data.name}_ch.png', bbox_inches='tight')
        # plt.clf()

        # plt.plot(n_clusters, dbs)
        # plt.title("Davies Bouldin Scores", fontsize=20)
        # plt.xticks(n_clusters)
        # plt.xlabel("N. of clusters")
        # plt.ylabel("Score")
        # plt.savefig(f'out/km_{self.data.name}_db.png', bbox_inches='tight')
        # plt.clf()

        max = np.argmax(sils)
        if self.data.name == 'heart':
            best_k = 3
        else:
            best_k = 4
        self.k = best_k

        return best_k

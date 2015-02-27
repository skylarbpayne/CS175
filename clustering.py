from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def complete_linkage_clustering(Xtr: np.ndarray, k: int) -> np.ndarray:
    '''
    Performs hierarchical clustering with complete linkage

    :param Xtr: the observation x feature matrix
    :param k: the number of clusters
    
    :return: the cluster assignments for each data point
    '''
    complete = AgglomerativeClustering(n_clusters = k, linkage='complete')
    return iomplete.fit_predict(Xtr.toarray())

def ward_linkage_clustering(Xtr: np.ndarray, k: int) -> np.ndarray:
    '''
    Performs hierarchical clustering with complete linkage

    :param Xtr: the observation x feature matrix
    :param k: the number of clusters

    :return: the cluster assignments for each data point
    '''

    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    return ward.fit_predict(Xtr.toarray())

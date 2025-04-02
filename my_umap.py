from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class my_UMAP:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.a = p[0]
        b = p[1]

    def prob_low_dim(Y):
        """
        Compute matrix of probabilities q_ij in low-dimensional space
        """
        inv_distances = np.power(1 + np.square(euclidean_distances(Y, Y)), -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    def prob_low_dim(Y):
        """
        Compute matrix of probabilities q_ij in low-dimensional space
        """
        inv_distances = np.power(1 + a * np.square(euclidean_distances(Y, Y)) ** b, -1)
        return inv_distances

    def CE(P, Y):
        """
        Compute Cross-Entropy (CE) from matrix of high-dimensional probabilities
        and coordinates of low-dimensional embeddings
        """
        Q = prob_low_dim(Y)
        return - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)

    def CE_gradient(P, Y):
        """
        Compute the gradient of Cross-Entropy (CE)
        """
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        inv_dist = np.power(1 + a * np.square(euclidean_distances(Y, Y)) ** b, -1)
        Q = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q, axis=1, keepdims=True)
        fact = np.expand_dims(a * P * (1e-8 + np.square(euclidean_distances(Y, Y))) ** (b - 1) - Q, 2)
        return 2 * b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis=1)
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances

class my_UMAP:
    def __init__(self, learning):
        self.learning = learning

    def get_prob(self, A, x_i, x_j, pi, sigma_i, knn):
        if x_j in knn[x_i]:
            return np.exp(-(np.linalg.norm(A[x_i] - A[x_j]) - pi) / sigma_i)
        return 0

    def bin_search(self, arr, goal, epsilon):
        start, end = 0, len(arr) - 1
        for i in range(end + 1):
            middle = int((end - start) / 2)
            if goal - epsilon <= arr[middle] <= goal + epsilon:
                return middle
            elif arr[middle] < goal - epsilon:
                start = middle + 1
            else:
                end = middle - 1
        return -1

    def find_sigma_i(self, x_i, pi: float, k: int, Neighbours, A: np.ndarray):
        sig_i = np.linspace(0, 10, 100)
        goal = np.log2(k)

        def sig_exp(x_i, pi, sig, Neighbours, A):
            res = 0
            for x_ij in Neighbours:
                res += np.exp(-(np.linalg.norm(A[x_i] - A[x_ij]) - pi) / sig)
            return res

        sigs = sig_exp(x_i, pi, sig_i, Neighbours, A)
        sig = self.bin_search(sigs, goal, 0.01)
        return sig_i[sig]



    def prob_low_dim(Y):
        """
        Compute matrix of probabilities q_ij in low-dimensional space
        """
        inv_distances = np.power(1 + np.square(euclidean_distances(Y, Y)), -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    def prob_low_dim(self, Y):
        inv_distances = np.power(1 + a * np.square(euclidean_distances(Y, Y)) ** b, -1)
        return inv_distances

    def Cross_Entropy(P, Y):
        Q = self.prob_low_dim(Y)
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
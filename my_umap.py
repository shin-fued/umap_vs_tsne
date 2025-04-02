import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances

#tried to implement it, it was bad took this dude's: https://github.com/NikolayOskolkov/HowUMAPWorks/blob/master/HowUMAPWorks.ipynb and adapted

class my_UMAP:
    def __init__(self, learning):
        self.learning = learning

    def get_prob(self, A, x_i, x_j, pi, sigma_i, knn):
        if x_j in knn[x_i]:
            return np.exp(-(np.linalg.norm(A[x_i] - A[x_j]) - pi) / sigma_i)
        return 0

    def k(self, prob):
        """
        Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
        """
        return np.power(2, np.sum(prob))

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


    def prob_high_dim(self, rho, sigma, dist, dist_row):
        d = dist[dist_row]-rho[dist_row]
        d[d < 0] = 0
        return np.exp(- d / sigma)

    def sigma_binary_search(self, k_of_sigma, fixed_k):
        """
        Solve equation k_of_sigma(sigma) = fixed_k
        with respect to sigma by the binary search algorithm
        """
        sigma_lower_limit = 0
        sigma_upper_limit = 1000
        approx_sigma = 0
        for i in range(20):
            approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
            if k_of_sigma(approx_sigma) < fixed_k:
                sigma_lower_limit = approx_sigma
            else:
                sigma_upper_limit = approx_sigma
            if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
                break
        return approx_sigma

    def prob_low_dim(self, Y, a, b):
        """
            Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
            """
        inv_distances = np.power(1 + a * np.square(euclidean_distances(Y, Y)) ** b, -1)
        return inv_distances

    def Cross_Entropy(self, P, Y, a, b):
        """
    Compute Cross-Entropy (CE) from matrix of high-dimensional probabilities
    and coordinates of low-dimensional embeddings
    """
        Q = self.prob_low_dim(Y, a, b)
        return - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)

    def CE_gradient(self, P, Y, a, b):
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

    def a_b(self, MIN_DIST):
        x = np.linspace(0, 3, 300)

        def f(x, min_dist):
            y = []
            for i in range(len(x)):
                if (x[i] <= min_dist):
                    y.append(1)
                else:
                    y.append(np.exp(- x[i] + min_dist))
            return y

        dist_low_dim = lambda x, a, b: 1 / (1 + a * x ** (2 * b))

        p, _ = optimize.curve_fit(dist_low_dim, x, f(x, MIN_DIST))

        a = p[0]
        b = p[1]
        return a, b

    def get_CE(self):
        return self.CE

    def fit(self, X, neighbours, MIN_DIST, d, iter):
        n = X.shape[0]
        dist = np.square(euclidean_distances(X, X))
        rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
        prob = np.zeros((n, n))
        for dist_row in range(n):
            func = lambda sigma: self.k(self.prob_high_dim(rho=rho, sigma=sigma, dist=dist, dist_row=dist_row))
            binary_search_result = self.sigma_binary_search(func, neighbours)
            prob[dist_row] = self.prob_high_dim(rho =rho,sigma=binary_search_result, dist=dist, dist_row=dist_row)
        P = (prob + np.transpose(prob)) / 2
        np.random.seed(12345)
        model = SpectralEmbedding(n_components=d, n_neighbors=neighbours)
        y = model.fit_transform(np.log(X + 1))
        a,b = self.a_b(MIN_DIST)
        CE_array = []
        for i in range(iter):
            y = y - self.learning * self.CE_gradient(P=P, Y=y, a=a, b=b)
            CE_current = np.sum(self.Cross_Entropy(P, y, a, b)) / 1e+5
            CE_array.append(CE_current)
        self.CE = CE_array[-1]
        return y


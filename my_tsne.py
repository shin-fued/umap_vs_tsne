import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

class my_TSNE:
    def __init__(self, learning, moment):
        self.learning = learning
        self.moment = moment


    def prob_high_dim(self, sigma, dist_row, dist):
        exp_dist = np.exp(-dist[dist_row]/2*sigma**2)
        exp_dist[dist_row] = 0
        prob_not_sym = exp_dist/np.sum(exp_dist)
        return prob_not_sym

    def perplexity(self, prob):
        return np.power(2, -np.sum([p*np.log2(p) for p in prob if p!=0]))

    def sigma_binary_search(self, sigma_perp, fixed_perplexity):
        sigma_lower_limit = 0
        sigma_upper_limit = 1000
        approx_sigma = 0
        for i in range(20):
            approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
            if sigma_perp(approx_sigma) < fixed_perplexity:
                sigma_lower_limit = approx_sigma
            else:
                sigma_upper_limit = approx_sigma
            if np.abs(fixed_perplexity - sigma_perp(approx_sigma)) <= 1e-5:
                break
        return approx_sigma

    def prob_low_dim(self, Y):
        inv_distances = np.power(1 + np.square(euclidean_distances(Y, Y)), -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    def KL(self, P, Q):
        return P * np.log(P+0.01) - P * np.log(Q+0.01)

    def KL_grad(self, P, Y):
        Q = self.prob_low_dim(Y)
        ydiff = np.expand_dims(Y, axis=1) - np.expand_dims(Y, axis=1)
        inv_dist = np.power(1 + np.square(ydiff), -1)

    def get_sigma(self):
        return self.sigma_array




    def fit(self, X, Y, Perplexity, d, Iter):
        X = np.log(X + 1)
        dist = np.square(euclidean_distances(X, squared=True))
        n = X.shape[0]
        prob = np.zeros((n, n))
        sigma_array = []
        for dist_row in range(n):
            func = lambda sigma: self.perplexity(self.prob_high_dim(sigma, dist_row))
            binary_search_result = self.sigma_binary_search(func, Perplexity)
            prob[dist_row] = self.prob_high_dim(binary_search_result, dist_row)
            sigma_array.append(binary_search_result)
        self.sigma_array = sigma_array
        P = prob + np.transpose(prob)
        Q = self.prob_low_dim(Y)
        y = np.random.normal(loc = 0, scale = 1e4, size=(n, d))
        KL_array = []





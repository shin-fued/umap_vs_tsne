import sklearn
import scipy.optimize as opt

class umap:
    def __init__(self, data):
        self.data = data

    def train(self):
        data = self.data
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')



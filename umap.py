import sklearn
import scipy.optimize as opt
from numba import njit
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame


#input in as a pandas dataframe
@njit
def umap(data: DataFrame):
    return



dt = pd.read_csv("diabetes.csv")
umap()
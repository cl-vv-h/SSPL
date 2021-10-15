from sklearn.linear_model import LinearRegression as LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as ssp
from scipy.optimize import nnls

def WGC(source, target, k):

    def weight_vector(xi, tree, k):
        Nx,Knn_index = KNN_find(xi, tree, k)
        NxT = np.transpose(Nx)
        w = np.zeros(source.shape[0])
        Para = np.array(nnls(NxT,xi))
        for i in range(0,k):
            w[Knn_index[i]] = Para[0][i]
        return w

    def KD(data):
        return ssp.KDTree(data)

    def KNN_find(xi, tree, k):
        KNN_index = tree.query(xi, k=k)[1]
        KNN_list = []
        for i in KNN_index:
            KNN_list.append(tree.data[i])
        return np.array(KNN_list),KNN_index

    tree_s = KD(source)
    #print(KNN_find(target[0],tree_s,10)[1])
    w_list = []
    for xi in target:
        wi = weight_vector(xi,tree_s,k)
        w_list.append(wi)
    
    return np.array(w_list)



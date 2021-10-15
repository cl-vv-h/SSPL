from sklearn.linear_model import LinearRegression as LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as ssp
from scipy.optimize import leastsq as lq
from WGC import WGC
import scipy.io as scio
from scipy.optimize import least_squares as lsq
from scipy.optimize import nnls



data = scio.loadmat('./sampleData.mat')

partialData = data.get('partialData')
unlabeledData = data.get('unlabeledData')
partialTarget = data.get('partialTarget')

partialData = np.array(partialData)
unlabeledData = np.array(unlabeledData)
#partialTarget = np.array(partialTarget)




a,size = partialTarget.shape
tree = ssp.KDTree(partialData)
asd = partialData.shape[0]
def weight_vector(xi, tree, k,asd):
    Nx,Knn_index = KNN_find(xi, tree, k)
    NxT = np.transpose(Nx)
    print(Knn_index)
    w = np.zeros(asd)
    Para = np.array(nnls(NxT,xi))
    for i in range(0,k):
        w[Knn_index[i]] = Para[0][i]
    return w

def KNN_find(xi, tree, k):
    KNN_index = tree.query(xi, k=k)[1]
    KNN_list = []
    for i in KNN_index:
        KNN_list.append(tree.data[i])
    return np.array(KNN_list),KNN_index

t = weight_vector(unlabeledData[3,:],tree,10,asd)

np.savetxt('vector.csv',t,delimiter=',')


#print(testData)
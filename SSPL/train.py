from sklearn.linear_model import LinearRegression as LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as ssp
from scipy.optimize import nnls
from WGC import WGC
import scipy.io as scio
import math


def propagation(Xu,Xp,Sp,k,alpha,beta,MaxIter):
    sums = Sp.sum(axis=1)

    P = Sp/sums[:,None]
    Fp = P
    Fu = np.ones((Xu.shape[0],Sp.shape[1]))/Sp.shape[1]
    H = WGC(Xp,Xu,k) 
    np.savetxt('H.csv',H,delimiter=',')
    J = WGC(Xp,Xp,k) 
    np.savetxt('J.csv',J,delimiter=',')
    K = WGC(Xu,Xu,k) 
    np.savetxt('K.csv',K,delimiter=',')
    L = WGC(Xu,Xp,k) 
    np.savetxt('L.csv',L,delimiter=',')

    for i in range(MaxIter):
        print(i)
        tmp = np.copy(Fp)
        if i <= 50:
            beta_ = (i/MaxIter) * beta
        else:
            beta_ = beta
        Fu = alpha * np.dot(H,Fp) + (1-alpha) * Fu
        Fp = alpha * np.dot(J,Fp) + (1-alpha) * P

        Fp = Fp*Sp
        Fp = Fp/(Fp.sum(axis=1)[:,None])

        Fu = beta_ * np.dot(K,Fu) + (1-beta_) * Fu
        Fp = beta_ * np.dot(L,Fu) + (1-beta_) * Fp

        Fp = Fp*Sp
        Fp = Fp/(Fp.sum(axis=1)[:,None])

        diff = np.linalg.norm((Fp-tmp)).sum()

        if diff<0.0001 and i>=1:
            break

    return Fp,Fu


def confi_matrix(sets_label, partialTarget):
    confi_matrix = np.zeros((sets_label.shape),dtype=int)
    for i in range(sets_label.shape[0]):
        y = np.argmax(sets_label[i])
        confi_matrix[i,y] = 1

    return confi_matrix


def confi_matrix1(sets_label, partialTarget):
    confi_matrix = np.zeros((sets_label.shape),dtype=int)
    nc_ = np.sum(partialTarget,axis=0)
    nc_hat = np.sum(sets_label,axis=0)
    for i in range(sets_label.shape[0]):
        yi = (nc_ / nc_hat) * sets_label[i]
        yi_hat = np.argmax(yi)
        confi_matrix[i,yi_hat] = 1

    return confi_matrix

def predict(x_star,confi_matrix,partialData,unlabeledData,k,r):
    w_p,index_p = weight_vector(x_star, partialData,k)
    w_u,index_u = weight_vector(x_star, unlabeledData,k)
    X_p = partialData[index_p,:]
    X_u = unlabeledData[index_u,:]
    q = confi_matrix.shape[1]
    l = partialData.shape[1]
    Y_star = np.zeros((q,))
    for i in range(q):
        x_knn_p = np.zeros((l,))
        x_knn_u = np.zeros((l,))
        for j in range(k):
            if np.argmax(confi_matrix[index_p[j]])==i:
                x_knn_p = x_knn_p + r * w_p[j] * X_p[j]
            
            if np.argmax(confi_matrix[index_u[j]])==i:
                x_knn_u = x_knn_u - (1-r) * w_u[j] * X_u[j]
        
        Y_star[i] = np.linalg.norm((x_star-x_knn_p-x_knn_u))
    
    return np.argmax(Y_star)



def KD(data):
    return ssp.KDTree(data)


def KNN_find(xi, tree, k):
    KNN_index = tree.query(xi, k=k)[1]
    KNN_list = []
    for i in KNN_index:
        KNN_list.append(tree.data[i])
    return np.array(KNN_list),KNN_index


def weight_vector(xi, data, k):
    tree = KD(data)
    Nx,Knn_index = KNN_find(xi, tree, k)
    NxT = np.transpose(Nx)
    w = np.zeros(data.shape[0])
    Para = np.array(nnls(NxT,xi))
    for i in range(0,k):
        w[Knn_index[i]] = Para[0][i]
    return Para[0],Knn_index


data = scio.loadmat('./sampleData.mat')

partialData = data.get('partialData')
unlabeledData = data.get('unlabeledData')
partialTarget = data.get('partialTarget').toarray()
a,size = data.get('partialTarget').shape



k=10
r=0.7
alpha=0.8

beta=0.25
Fp,Fu = propagation(unlabeledData,partialData,partialTarget,10,alpha,beta,100)



np.savetxt('py_Fp.csv', Fp, delimiter=',')
np.savetxt('py_Fu.csv', Fu, delimiter=',')



setU = np.hstack((unlabeledData,Fu))
setP = np.hstack((partialData,Fp))

sets_Data = np.vstack((partialData,unlabeledData))
sets_label = np.vstack((Fp,Fu))

confi_matrix = confi_matrix(sets_label,partialTarget)

testData = data.get('testData')
testTarget = data.get('testTarget')


count = 0.0
num = testData.shape[0]

for i in range(num):
    a = predict(testData[i],confi_matrix, partialData, unlabeledData, k, r)
    if a==np.argmax(testTarget[i]):
        count = count + 1

acc = count/num

print(acc)




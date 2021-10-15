import scipy.io as scio

data = scio.loadmat('./sampleData.mat')
print(data.get('partialData'))

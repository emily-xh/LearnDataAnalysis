# Target: Visualize the data based on specified feature, color the point by the saling price
# Method: Using the Principal Component Analysis
#
from DataLoader import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm

def Visualization(X, Y):
    if X.shape[1] > 2:
        X = PCA(n_components=2).fit(X.astype('float32'))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    Y = np.round(100 * Y).astype('int16').reshape(-1)
    c = cm.rainbow(np.linspace(0,1,np.max(Y)+1))
    plt.scatter(X[:, 0], X[:, 1], c=c[Y], s=1)
    plt.show()

feature_list = ['Street','Alley','Condition1']
target_list = ['SalePrice']

if __name__ == "__main__":
    pos, data = CSVLoader('./data/train.csv')
    x_train = getAttribute(feature_list, pos, data)
    y_train = getAttribute(target_list, pos, data)
    print('#Training samples=%d'%len(x_train))
    print('Training featrue dimansion=%d'%len(x_train[0]))
    Visualization(x_train, y_train)

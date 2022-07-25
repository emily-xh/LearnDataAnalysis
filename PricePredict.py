# Target: predict the selling price based on some specified feature attributes
# Method: Linear Regression (affine)
#           AX+b = Y 
#       find the optimal A, b using the given training data
#       for testing, using the derived A, b to predict the price
#           X is feature attributes, shape (n_samples, n_features)
# Evaluation: Root Mean Square Error (RMSE)
#               RMSE = sqrt(mean((Y-pY)^2))
#
import numpy as np
from DataLoader import *

class LinearRegression():
    def __init__(self):
        self.weight = []

    def fit(self, X, Y):
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        self.weight, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def predict(self, X):
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        pred = np.matmul(X, self.weight)
        return pred

def RMSE(y, py):
    return np.sqrt(np.mean(np.square(y-py)))

def PrintPrediction(pred, feature_list, position_dict, data, TrueValue=None):
    for key in feature_list:
        print(key,data[position_dict[key]], end=' | ')
    print()
    if TrueValue is not None:
        print("predicted price", pred, 'True price', TrueValue)
    else:
        print("predicted price", pred)

feature_list = ['Street','Alley','Condition1']
target_list = ['SalePrice']

if __name__ == "__main__":
    pos, data = CSVLoader('./data/train.csv')
    x_train = getAttribute(feature_list, pos, data)
    y_train = getAttribute(target_list, pos, data)
    print('#Training samples=%d'%len(x_train))
    print('Training featrue dimansion=%d'%len(x_train[0]))
    lr = LinearRegression().fit(x_train, y_train)
    py_train = lr.predict(x_train)
    print('LinearRegression fitting RMSE %f'%RMSE(y_train, py_train))
    print('First 10 Predicted SalePrice on Testing set...')
    for i in range(10):
        PrintPrediction(py_train[i], feature_list, pos, data[i], y_train[i])

    print('---------------')
    print("Testing...")
    post, datat = CSVLoader('./data/test.csv')
    x_test = getAttribute(feature_list, post, datat)
    print('#Testing samples=%d'%len(x_test))
    print('Testing featrue dimansion=%d'%len(x_test[0]))
    py_test = lr.predict(x_test)
    print('First 10 Predicted SalePrice on Testing set...')
    for i in range(10):
        PrintPrediction(py_test[i], feature_list, post, datat[i])

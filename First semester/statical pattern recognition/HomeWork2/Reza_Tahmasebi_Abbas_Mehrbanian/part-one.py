import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import scipy.stats as stats

data = np.genfromtxt('seed.txt', delimiter='\t', skip_header=0)

X = data[:,:-1]
y = data[:,-1:]
y = np.select([y == 1, y == 2, y == 3], [0, 1, 2], y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

m = len(X_train)
X_train = np.hstack((np.ones((m,1)),X_train))
X_test = np.hstack((np.ones((len(X_test), 1)), X_test))

def cal_normal_dist(X):
    h = np.array(X)
    h.sort()
    hmean = np.mean(h)
    hstd = np.std(h)
    pdf = stats.norm.pdf(h, hmean, hstd)
    return h, pdf

y_ = y.flatten()

fig, axs = plt.subplots(7,7, figsize=(30,30))
for i in range(7):
    for j in range(7):
        if (i == j):
            h = np.array(X[y_ == 0, i])
            h.sort()
            hmean = np.mean(h)
            hstd = np.std(h)
            pdf = stats.norm.pdf(h, hmean, hstd)
            h1, pdf1 = cal_normal_dist(X[y_ == 0, i])
            h2, pdf2 = cal_normal_dist(X[y_ == 1, i])
            h3, pdf3 = cal_normal_dist(X[y_ == 2, i])
            axs[i][j].plot(h1, pdf1, c='b') 
            axs[i][j].plot(h2, pdf2, c='r')
            axs[i][j].plot(h3, pdf3, c='g')
            continue
        axs[i][j].scatter(X[y_ == 0, i], X[y_ == 0, j], s=15, marker='o', c='b', label='Class 1')
        axs[i][j].scatter(X[y_ == 1, i], X[y_ == 1, j], s=15, marker='x', c='r', label='Class 2')
        axs[i][j].scatter(X[y_ == 2, i], X[y_ == 2, j], s=15, marker='x', c='g', label='Class 2')
        axs[i][j].set_xlabel(('Feature no.' + str(i+1)))
        axs[i][j].set_ylabel(('Feature no.' + str(j+1)))


def sigmoid(x):    
    return 1 / (1 + np.exp(-x))

def compute_loss(y, y_pred):
    y_zero_loss = y * np.log(y_pred)
    y_one_loss = (1-y) * np.log(1 - y_pred)
    return -np.mean(y_zero_loss + y_one_loss)

def one_vs_one_data(X, y, i, j):
    X_ = X
    y_ = y
    X_ = X_[np.logical_or(y == i, y == j), :]
    y_ = y_[np.logical_or(y == i, y == j)]
    y_[y_ == i] = 0
    y_[y_ == j] = 1
    return X_, y_

def one_vs_all_data(y, vsclass):
    y_ = np.select([y == vsclass, y != vsclass], [1, 0], y)
    return y_

def binary_classifier_model(x, y, alpha, weights, iterations): 
    m = x.shape[0]
    loss_history = []
    # convergance_point = None
    for i in range(iterations): 
        y_pred = sigmoid(np.dot(x, weights))
        weights = weights - alpha * (1/m * np.dot(x.T, (y_pred - y)))

        loss = -1/m * np.sum((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)), axis = 0)
        # if(i != 0 and (convergance_point is None) and abs(loss - loss_history[-1]) < 0.00005 * loss_history[-1]):
        #     convergance_point = i
        loss_history.append(loss)
    
    return weights, loss_history


def get_accuracy(x, y, weights):  
    probs = sigmoid(np.dot(x, weights))
    preds = np.argmax(probs, axis = 1)

    accuracy = sum(preds == y.flatten())/(float(len(y)))
    return accuracy

iterations = 15000
alpha = 0.01
weights = []
loss_log = []
xi_s = []

for j in range(3):
    for i in range(j):
        X_ij_train, y_ij_train = one_vs_one_data(X_train, y_train.flatten(), i, j)
        weights_ij = np.zeros(X_ij_train.shape[1])
        weights_ij, loss_history_ij = binary_classifier_model(X_ij_train, y_ij_train, alpha, weights_ij, iterations)
        weights.append(np.array(weights_ij))
        loss_log.append(loss_history_ij)

loss_log = np.array(loss_log).T
plt.figure(figsize=(7,5))
plt.plot(list(range(len(loss_log))), loss_log[:,0], 'b', label='class 1 vs class 2')
plt.plot(list(range(len(loss_log))), loss_log[:,1], 'r', label='class 1 vs class 3')
plt.plot(list(range(len(loss_log))), loss_log[:,2], 'g', label='class 2 vs class 3')
plt.plot(list(range(len(loss_log))), loss_log.mean(1), linestyle = 'dashed' , color = 'black' , label='MEAN')
plt.title('Cost / Iterations')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.legend(loc = 'upper right')
plt.show()

def predict_labels(X, weights):
  labels = []
  for i in range(len(X)):
    cout_preds_class = [0, 0, 0]
    pred1 = pred2 = pred3 = False
    pred1 = sigmoid(np.dot(X[i], weights[0])) >= 0.5
    pred2 = sigmoid(np.dot(X[i], weights[1])) >= 0.5
    pred3 = sigmoid(np.dot(X[i], weights[2])) >= 0.5

    # pred1 is for class 0 vs 1
    # cout_preds_class[1] = cout_preds_class[1] + cout_preds_class(pred1)
    # cout_preds_class[0] = cout_preds_class[0] + cout_preds_class(pred1)
    if pred1:
      cout_preds_class[1] = cout_preds_class[1] + 1
    else:
      cout_preds_class[0] = cout_preds_class[0] + 1
    # pred2 is for class 0 vs 2
    if pred2:
      cout_preds_class[2] = cout_preds_class[2] + 1
    else:
      cout_preds_class[0] = cout_preds_class[0] + 1
    # pred3 is for class 1 vs 2
    if pred3:
      cout_preds_class[2] = cout_preds_class[2] + 1
    else:
      cout_preds_class[1] = cout_preds_class[1] + 1
    labels.append(cout_preds_class)
  return [l.index(max(l)) for l in labels]

y_pred = predict_labels(X_train, np.array(weights))
train_accuracy = sum(y_pred == y_train.flatten())/(float(len(y_train)))

y_pred_test = predict_labels(X_test, np.array(weights))
test_accuracy = sum(y_pred_test == y_test.flatten())/(float(len(y_test)))

print('Training Accuracy (One-vs-One): ', train_accuracy)
print('Test Accuracy (One-vs-One): ', test_accuracy)


# 1 vs all
iterations = 20000
alpha = 0.015
m = len(y_train)

# method 1

#For multi class classification we uses the concept of one vs all
#as we have 3 classes so y will be of shape m,3
Y_train = np.zeros((m,3))
clasess = [0, 1, 2]
#in Y matrix for each class columns put 1 wherin the rows belong to that class
for cls in clasess:
    Y_train[np.where(y_train[:,-1] == cls), cls] = 1

weights = np.zeros((X_train.shape[1], Y_train.shape[1]), dtype='float64')

(weights, loss_log) = binary_classifier_model(X_train, Y_train, alpha, weights, iterations)

loss_log = np.array(loss_log)
plt.figure(figsize=(7,5))
plt.plot(list(range(len(loss_log))), loss_log[:,0], 'b', label='Model 1 v ALL')
plt.plot(list(range(len(loss_log))), loss_log[:,1], 'r', label='Model 2 v ALL')
plt.plot(list(range(len(loss_log))), loss_log[:,2], 'g', label='Model 3 v ALL')
plt.plot(list(range(len(loss_log))), loss_log.mean(1), linestyle = 'dashed' , color = 'black' , label='MEAN')
plt.legend(loc='best')
plt.show()

print('Training Accuracy (One-vs-All): ', get_accuracy(X_train, y_train, weights))
print('Test Accuracy (One-vs-All): ', get_accuracy(X_test, y_test, weights))

# method 2
weights = []
loss_log = []

for i in range(3):
    y_i_train = one_vs_all_data(y_train.flatten(), i)
    weights_i = np.zeros(X_train.shape[1])
    weights_i, loss_history_i = binary_classifier_model(X_train, y_i_train, alpha, weights_i, iterations)
    weights.append(weights_i)
    loss_log.append(loss_history_i)

loss_log = np.array(loss_log).T
plt.figure(figsize=(7,5))
plt.plot(list(range(len(loss_log))), loss_log[:,0], 'b', label='Model 1 v ALL')
plt.plot(list(range(len(loss_log))), loss_log[:,1], 'r', label='Model 2 v ALL')
plt.plot(list(range(len(loss_log))), loss_log[:,2], 'g', label='Model 3 v ALL')
plt.plot(list(range(len(loss_log))), loss_log.mean(1), linestyle = 'dashed' , color = 'black' , label='MEAN')
plt.legend(loc='best')
plt.show()

print('Training Accuracy (One-vs-All): ', get_accuracy(X_train, y_train, np.array(weights).T))
print('Test Accuracy (One-vs-All): ', get_accuracy(X_test, y_test, np.array(weights).T))


# softmax
def softmax(z):
    z -= np.max(z) # for stability
    return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T


def softmax_train(x, y, alpha, weights, iterations): 
    m = x.shape[0]
    loss_history = []
    for i in range(iterations): 
        y_pred = softmax(np.dot(x, weights))

        weights = weights - alpha * (1/m * np.dot(x.T, (y_pred - y)))
        loss = -1/m * np.sum((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)), axis = 0)
        # loss = loss.sum(axis = 0)
        loss_history.append(loss)
    
    return weights, loss_history

    
def get_softmax_accuracy(x, y, weights):  
    probs = softmax(np.dot(x, weights))
    preds = np.argmax(probs, axis = 1)

    accuracy = sum(preds == y.flatten())/(float(len(y)))
    return accuracy


weights = np.zeros((X_train.shape[1], Y_train.shape[1]), dtype='float64')
iterations = 15000
alpha = 0.01
m = len(y_train)

(weights, loss_history) = softmax_train(X_train, Y_train, alpha, weights, iterations)

loss_log = np.array(loss_history)
plt.figure(figsize=(7,5))
plt.plot(list(range(len(loss_log))), loss_log[:,0], 'b', label='Model 1 v ALL')
plt.plot(list(range(len(loss_log))), loss_log[:,1], 'r', label='Model 2 v ALL')
plt.plot(list(range(len(loss_log))), loss_log[:,2], 'g', label='Model 3 v ALL')
plt.plot(list(range(len(loss_log))), loss_log.mean(1), linestyle = 'dashed' , color = 'black' , label='MEAN')
plt.legend(loc='best')
plt.show()

print('Training Accuracy (Softmax): ', get_softmax_accuracy(X_train, y_train, weights))
print('Test Accuracy (Softmax): ', get_softmax_accuracy(X_test, y_test, weights))



# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading data
data = pd.read_csv("data.csv", header=None, names=['x','y'])
train_data, test_data = train_test_split(data, test_size=0.3)

X_train = np.array(train_data['x'])
y_train = np.array(train_data['y'])
X_test = np.array(test_data['x'])
y_test = np.array(test_data['y'])
from utils import normalize, add_bias
# normalizing x train
X_train_norm = normalize(X_train)
# adding bias (=1) as a feature
X_train_norm_bias = add_bias(X_train_norm)

# normalizing y train
y_train_norm = normalize(y_train)

# same things for test data
X_test_norm = normalize(X_test)
X_test_norm_bias = add_bias(X_test_norm)
y_test = normalize(y_test)
def cal_cost(y, y_pred):
    m = y.shape[0]
    return (1/m) * np.sum((y_pred - y)**2)

theta = np.random.rand(2, 1)
iterations = 1200
alpha = 0.05

def batch_gradient_descent(X, y, theta, alpha, iterations):
    theta_history = np.zeros((iterations, 2))
    cost_history = np.zeros(iterations)
    m = len(y_train_norm)
    
    for i in range(iterations):
        y_pred = X.dot(theta)
        theta = theta - (1 / m) * alpha * X.T.dot((y_pred - y))
        cost = cal_cost(y, y_pred)
        theta_history[i, :] = theta.T
        cost_history[i] = cost

    return theta, cost_history, theta_history

(theta, costs_h, theta_h) = batch_gradient_descent(X_train_norm_bias, y_train_norm, theta, alpha, iterations)

from IPython.display import Latex

y_pred = X_train_norm_bias.dot(theta)
cost_train = cal_cost(y_train_norm, y_pred)

plt.figure(figsize=(7,5))
plt.plot(X_train, y_train_norm, "ko", markersize=1)
plt.plot(X_train, y_pred, "r-", linewidth=1)
plt.title("Train Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
epsilon_c = 0.003
epsilon_y = 0.005


print('The parameter values (theta0 & theta1): {} and {}'.format(theta.flatten()[0], theta.flatten()[1]))
print('Train cost(mse): {}'.format(cost_train))
print('Decision boundry with the format of "y = ax + b":')
Latex(f"""\\begin{{equation*}}
\\hat{{y}} = {{{theta.flatten()[1]}}}x_1 + {{{theta.flatten()[0]}}}
\\end{{equation*}}
""")
y_pred = X_test_norm_bias.dot(theta)
cost_test = cal_cost(y_test, y_pred)

plt.figure(figsize=(7,5))
plt.plot(X_test, y_test, "ko", markersize=1)
plt.plot(X_test, y_pred, "r-", linewidth=1)
plt.title("Test Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print('Test cost(mse): {}'.format(cost_test))
print('Decision boundry with the format of "y = ax + b":')
Latex(f"""\\begin{{equation*}}
\\hat{{y}} = {{{theta.flatten()[1]}}}x_1 + {{{theta.flatten()[0]}}}
\\end{{equation*}}
""")
plt.figure(figsize=(7,5))
plt.plot(range(iterations), costs_h, 'b.', markersize = 2)
plt.title('Cost / Iterations')
plt.ylabel('Cost (MSE)')
plt.xlabel('Iterations')
plt.show()


theta = np.random.rand(2, 1)
epochs = 1200
alpha = 0.05
batch_size = 25

def mini_batch_gradient_descent(X, y, theta, alpha, epochs, batch_size):
    theta_history = np.zeros((epochs, 2))
    cost_history = np.zeros(epochs)
    m = len(y)
    batches_num = int(m/batch_size)

    for i in range(epochs):
        cost = 0
        for j in range(0, m, batch_size):
            X_j = X[j:j + batch_size]
            y_j = y[j:j + batch_size]
            y_pred_j = X_j.dot(theta)
            theta = theta - (1 / m) * alpha * X_j.T.dot((y_pred_j - y_j))
            cost += cal_cost(y_j, y_pred_j)
        theta_history[i, :] = theta.T
        cost_history[i] = cost

    return theta, cost_history, theta_history

(theta, costs_h, theta_h) = mini_batch_gradient_descent(X_train_norm_bias, y_train_norm, theta, alpha, iterations, batch_size)
from IPython.display import Latex

y_pred = X_train_norm_bias.dot(theta)
cost_train = cal_cost(y_train_norm, y_pred)

plt.figure(figsize=(7,5))
plt.plot(X_train, y_train_norm, "ko", markersize=1)
plt.plot(X_train, y_pred, "r-", linewidth=1)
plt.title("Train Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print('The parameter values (theta0 & theta1): {} and {}'.format(theta.flatten()[0], theta.flatten()[1]))
print('Train cost(mse): {}'.format(cost_train))
print('Decision boundry with the format of "y = ax + b":')
Latex(f"""\\begin{{equation*}}
\\hat{{y}} = {{{theta.flatten()[1]}}}x_1 + {{{theta.flatten()[0]}}}
\\end{{equation*}}
""")
y_pred = X_test_norm_bias.dot(theta)
cost_test = cal_cost(y_test, y_pred)

plt.figure(figsize=(7,5))
plt.plot(X_test, y_test, "ko", markersize=1)
plt.plot(X_test, y_pred, "r-", linewidth=1)
plt.title("Test Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print('Test cost(mse): {}'.format(cost_test))
print('Decision boundry with the format of "y = ax + b":')
Latex(f"""\\begin{{equation*}}
\\hat{{y}} = {{{theta.flatten()[1]}}}x_1 + {{{theta.flatten()[0]}}}
\\end{{equation*}}
""")
plt.figure(figsize=(7,5))
plt.plot(range(iterations), costs_h, 'b.', markersize = 2)
plt.title('Cost / Iterations')
plt.ylabel('Cost (MSE)')
plt.xlabel('Iterations')
plt.show()
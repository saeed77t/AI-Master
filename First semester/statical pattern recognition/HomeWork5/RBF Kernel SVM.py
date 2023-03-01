# %%
import numpy as np
from mat4py import loadmat
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# %%
data = loadmat('Dataset1.mat')
data2 = loadmat('Dataset2.mat')
data.keys()

# %%
data.keys()
x = data['X']
y = data['y']
label = []
for i in range(len(y)):
    label.append(y[i][0])

datas = pd.DataFrame(x, columns=['x1', 'x2'])
datas['y'] = label

# %%
x2 = data2['X']
y2 = data2['y']
label2 = []
for i in range(len(y2)):
    label2.append(y2[i][0])

datas2 = pd.DataFrame(x2, columns=['x1', 'x2'])
datas2['y'] = label2

# %%


class PreprocessData:
    def NomalizeData(Data):
        Data = np.array(Data)
        lenght = len(Data)
        normalizedData = []

        for i in range(lenght):
            normalizedData.append(
                float((Data[i] - min(Data)) / (max(Data) - min(Data))))

        return normalizedData

    def TestAndTrain(Data, PercentageOfTrainData):
        PercentageOfTrainData = float(PercentageOfTrainData / 100)
        Train_DataFrame = Data.sample(frac=PercentageOfTrainData)
        Test_DataFrame = Data.drop(Train_DataFrame.index)

        return Train_DataFrame, Test_DataFrame


def k_fold(data_length, k):
    folds = []
    fold_size = data_length/k
    for i in range(k):
        folds.append([int(i*fold_size), int(((i+1)*fold_size)) - 1])
    return folds


def get_folds(data, k):
    return np.split(data, k)


# %%
# split data1
Dataset = PreprocessData.TestAndTrain(datas, 80)

TrainData = Dataset[0]
TestData = Dataset[1]
data = np.array(TrainData)
Y = TrainData['y']
Y = np.array(Y)
data = TrainData.drop('y', axis=1)
X = np.array(data)
y_test = np.array(TestData['y'])
X_test = np.array(TestData.drop('y', axis=1))

# kfold dataset 1

# split data2
Dataset2 = PreprocessData.TestAndTrain(datas2, 80)

TrainData2 = Dataset2[0]
TestData2 = Dataset2[1]
data2 = np.array(TrainData2)
Y2 = TrainData2['y']
Y2 = np.array(Y2)
data2 = TrainData2.drop('y', axis=1)
X2 = np.array(data2)
y_test2 = np.array(TestData2['y'])
X_test2 = np.array(TestData2.drop('y', axis=1))

# %%
# 10 fold of dataset 2
cp_data2 = datas2

cp_data2 = cp_data2.to_numpy()
np.random.shuffle(cp_data2)
cp_data2 = np.delete(cp_data2, 0, 0)
cp_data2 = np.delete(cp_data2, 10, 0)
cp_data2 = np.delete(cp_data2, 20, 0)
# print(len(cp_data2))
folds2 = get_folds(cp_data2, 10)


# %%
# 10 fold dataset 1
# 10 fold of dataset 2
cp_data1 = datas

cp_data1 = cp_data1.to_numpy()

np.random.shuffle(cp_data2)
cp_data1 = np.delete(cp_data1, 0, 0)

# print(len(cp_data2))
folds1 = get_folds(cp_data1, 10)

# %%
folds1 = np.array(folds1)

# %%
Y[Y == 0] = -1
Y2[Y2 == 0] = -1
y_test[y_test == 0] = -1
y_test2[y_test2 == 0] = -1

# %%


def sigmoid(z):
    return (1 / (1 + np.exp(-1*z)))


def hypotesis(x, w, b):
    z = b+w[0][0]*x[:, 0] + w[0][1]*x[:, 1]
    yhat = sigmoid(z)
    return yhat


def hypotesis1(x, w, b):
    z = b+w[0]*x[:, 0] + w[1]*x[:, 1]
    yhat = sigmoid(z)
    return yhat

# %%


def predict(x, y, w, b):
    pred = hypotesis(x, w, b)
    predicted = hypotesis(x, w, b)
    predicted[predicted > 0.5] = 1
    predicted[predicted <= 0.5] = -1
    acc2 = np.mean(y == predicted)
    print('Youre accuracy is  = ', acc2*100)

# %%


class SVM:
    def __init__(self, kernel='rbf', C=10.0, max_iteration=500, degree=3, sigma=1):
        self.kernel = {
            'rbf': lambda x, y: np.exp(-sigma*np.sum((y - x[:, np.newaxis])**2, axis=-1))}[kernel]
        self.C = C
        self.max_iter = max_iteration

    def CalcLoss(self, t, multiV, u):
        t = (np.clip(multiV + t*u, 0, self.C) - multiV)[1]/u[1]
        return (np.clip(multiV + t*u, 0, self.C) - multiV)[0]/u[0]

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y * 2 - 1
        self.lambdas = np.zeros_like(self.y, dtype=float)
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        for _ in range(self.max_iter):
            for indexes in range(len(self.lambdas)):
                RightColumn = np.random.randint(0, len(self.lambdas))
                checkRL = self.K[[[indexes, indexes], [RightColumn, RightColumn]], [
                    [indexes, RightColumn], [indexes, RightColumn]]]
                multiV = self.lambdas[[indexes, RightColumn]]
                summeddata = 1 - \
                    np.sum(self.lambdas *
                           self.K[[indexes, RightColumn]], axis=1)
                u = np.array([-self.y[RightColumn], self.y[indexes]])
                t_max = np.dot(summeddata, u) / \
                    (np.dot(np.dot(checkRL, u), u) + 1E-15)
                self.lambdas[[indexes, RightColumn]] = multiV + \
                    u * self.CalcLoss(t_max, multiV, u)

        lended, = np.nonzero(self.lambdas > 1E-15)
        self.b = np.mean(
            (1.0 - np.sum(self.K[lended] * self.lambdas, axis=1)) * self.y[lended])

    def decide(self, X):
        return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b

    def predict(self, X):
        return (np.sign(self.decide(X)) + 1) // 2


# %%
def test_plot(X, y, svm_model, axes, title):
    plt.axes(axes)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]

    xx, yy = np.meshgrid(np.linspace(*xlim, num=700),
                         np.linspace(*ylim, num=700))
    rgb = np.array([[210, 0, 0], [0, 0, 150]])/255.0

    svm_model.fit(X, y)
    z_model = svm_model.predict(
        np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Z = z_model.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.contour(xx, yy, z_model, colors='k',
                levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)),
                 alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
    # plt.contour(xx, yy, Z, cmap=plt.cm.gray, levels=[0.5])
    plt.title(title)


# %%
Y2[Y2 == -1] = 0
y_test2[y_test2 == -1] = 0
sv = SVM(C=0.1, sigma=0.1)
sv.fit(X2, Y2)
r = sv.decide(X2)
pred = sv.predict(X2)
np.mean(Y2 == pred)

# %%
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
Y2[Y2 == -1] = 0
y_test2[y_test2 == -1] = 0

# %%
result = []
count = 1
for c in Cs:
    for si in sigmas:
        sv = SVM(C=c, sigma=si)
        sv.fit(X2, Y2)
        pred = sv.predict(X2)
        acc = np.mean(Y2 == pred)
        testpred = sv.predict(X_test2)
        acctest = np.mean(y_test2 == testpred)
        objective = (acctest*100*1.5)+acc*100
        result.append([c, si, acc, acctest, objective])
        print(str(count)+'- C = '+str(round(c, 2)) + ' ,Sigma = ' + str(round(si, 2)) + ', train accuracy = ' +
              str(round(acc*100, 2))+' ,test accuracy = '+str(round(acctest*100, 2))+' ,objective =  ' + str(round(objective, 2)))
        count += 1

# %%
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
Y[Y == -1] = 0
y_test[y_test == -1] = 0

# %%
result1 = []
count = 1
for c in Cs:
    for si in sigmas:
        sv = SVM(C=c, sigma=si)
        sv.fit(X, Y)
        pred = sv.predict(X)
        acc = np.mean(Y == pred)
        testpred = sv.predict(X_test)
        acctest = np.mean(y_test == testpred)
        objective = (acctest*100*1.5)+acc*100
        result1.append([c, si, acc, acctest, objective])
        print(str(count)+'- C = '+str(round(c, 2)) + ' ,Sigma = ' + str(round(si, 2)) + ', train accuracy = ' +
              str(round(acc*100, 2))+' ,test accuracy = '+str(round(acctest*100, 2))+' ,objective =  ' + str(round(objective, 2)))
        count += 1

# %%
sv = SVM(C=40, sigma=40)
sv.fit(X2, Y2)
pred = sv.predict(X2)
acc = np.mean(Y2 == pred)
testpred = sv.predict(X_test2)
acctest = np.mean(y_test2 == testpred)
print('test accuracy  = '+str(acctest*100) + 'accuracy = '+str(acc*100))

# %%
# split data1
Dataset = PreprocessData.TestAndTrain(datas, 90)

TrainData = Dataset[0]
TestData = Dataset[1]
data = np.array(TrainData)
Y = TrainData['y']
Y = np.array(Y)
data = TrainData.drop('y', axis=1)
X = np.array(data)
y_test = np.array(TestData['y'])
X_test = np.array(TestData.drop('y', axis=1))

# split data2
Dataset2 = PreprocessData.TestAndTrain(datas2, 90)

TrainData2 = Dataset2[0]
TestData2 = Dataset2[1]
data2 = np.array(TrainData2)
Y2 = TrainData2['y']
Y2 = np.array(Y2)
data2 = TrainData2.drop('y', axis=1)
X2 = np.array(data2)
y_test2 = np.array(TestData2['y'])
X_test2 = np.array(TestData2.drop('y', axis=1))

# %%
# Cs = [0.01,0.03,0.1,0.3,1,3,10,30]
# sigmas = [0.01,0.03,0.1,0.3,1,3,10,30]
# Y2[Y2==-1] = 0
# y_test2[y_test2==-1] = 0

# %%
# result10 = []
# count = 1
# for c in  Cs:
#     for si in sigmas:
#         sv = SVM(C=c,sigma=si)
#         sv.fit(X2,Y2)
#         pred = sv.predict(X2)
#         acc = np.mean(Y2 == pred)
#         testpred = sv.predict(X_test2)
#         acctest = np.mean(y_test2 == testpred)
#         objective = (acctest*100*1.5)+acc*100
#         result10.append([c,si,acc,acctest,objective])
#         print(str(count)+' C = '+str(c) + ' ,Sigma = ' + str(si) + ', train accuracy = '+str(acc*100)+' ,test accuracy = '+str(acctest*100)+' ,objective =  ' +str(objective))
#         count +=1

# %%
sortedresult = sorted(result, key=lambda x: x[4], reverse=True)

# %%
bestC = sortedresult[1][0]
bestSigma = sortedresult[1][1]

# %%

sortedresult1 = sorted(result1, key=lambda x: x[4], reverse=True)
bestC1 = sortedresult1[1][0]
bestSigma1 = sortedresult1[1][1]

# %%
worstC = sortedresult[63][0]
WorstSigma = sortedresult[63][1]
worstC1 = sortedresult1[50][0]
worstSigma1 = sortedresult1[50][1]

# %%
folds2 = np.array(folds2)

# %%
KfoldResult2 = []
for i in range(len(folds2)):
    testdata = folds2[i]
    traindata = folds2[0]

    for j in range(len(folds2)):
        if(i != j):
            traindata = np.concatenate((traindata, folds2[j]))
    for d in range(traindata.shape[0]):
        if(d < 86):
            traindata = np.delete(traindata, d, 0)

    Xn = np.delete(traindata, 2, 1)
    Yn = traindata[:, 2]
    Xntest = np.delete(testdata, 2, 1)
    Yntest = testdata[:, 2]
    Yn[Yn == -1] = 0
    Yntest[Yntest == -1] = 0

    sv = SVM(C=bestC, sigma=bestSigma)
    sv.fit(Xn, Yn)
    pred = sv.predict(Xn)
    acc = np.mean(Yn == pred)
    testpred = sv.predict(Xntest)
    acctest = np.mean(Yntest == testpred)
    objective = (acctest*100*1.5)+acc*100
    KfoldResult2.append([acc, acctest])
    print('********')
    print('fold'+str(i+1)+' ==> 🛠train accuracy = '+str(round(acc*100, 2)
                                                        )+' ,🔬test accuracy = '+str(round(acctest*100, 2)))
    print('************')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    test_plot(Xn, Yn, SVM(C=bestC, sigma=bestSigma), axs[0], 'train  ')
    test_plot(Xntest, Yntest, SVM(C=bestC, sigma=bestSigma), axs[1], 'test')


# %%
KfoldResult1 = []
for i in range(len(folds1)):
    testdata = folds1[i]
    traindata = folds1[0]

    for j in range(len(folds1)):
        if(i != j):
            traindata = np.concatenate((traindata, folds1[j]))
    for d in range(traindata.shape[0]):
        if(d < 21):
            traindata = np.delete(traindata, d, 0)

    Xn = np.delete(traindata, 2, 1)
    Yn = traindata[:, 2]
    Xntest = np.delete(testdata, 2, 1)
    Yntest = testdata[:, 2]
    Yn[Yn == -1] = 0
    Yntest[Yntest == -1] = 0

    sv = SVM(C=bestC1, sigma=bestSigma1)
    sv.fit(Xn, Yn)
    pred = sv.predict(Xn)
    acc = np.mean(Yn == pred)
    testpred = sv.predict(Xntest)
    acctest = np.mean(Yntest == testpred)
    objective = (acctest*100*1.5)+acc*100
    KfoldResult1.append([acc, acctest])
    print('********')
    print('fold'+str(i+1)+' ==> 🛠train accuracy = '+str(round(acc*100, 2)
                                                        )+' ,🔬test accuracy = '+str(round(acctest*100, 2)))
    print('************')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    test_plot(Xn, Yn, SVM(C=bestC1, sigma=bestSigma1), axs[0], 'train  ')
    test_plot(Xntest, Yntest, SVM(C=bestC1, sigma=bestSigma1), axs[1], 'test')
    plt.show()

# %%
print('plot best and worst  of sigma and C OF Dataset 1 ')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
test_plot(X, Y, SVM(C=bestC, sigma=bestSigma), axs[0], 'BEST  ')
test_plot(X, Y, SVM(C=worstC, sigma=WorstSigma), axs[1], 'WORST')

# %%
print('plot best and worst  of sigma and C OF Dataset 2 ')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
test_plot(X2, Y2, SVM(C=bestC1, sigma=30), axs[0], 'BEST  ')
test_plot(X2, Y2, SVM(C=worstC1, sigma=worstSigma1), axs[1], 'WORST')

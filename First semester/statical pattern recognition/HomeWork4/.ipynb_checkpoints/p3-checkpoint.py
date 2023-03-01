# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# %%
olivetti = fetch_olivetti_faces(return_X_y=True)
a = fetch_olivetti_faces()

# %%
x = olivetti[0]
y = olivetti[1]

# %%
eigen_vectors = []
cnum = []

# %%


def CalculateScaters(x, y):
    NumberOfFeatures = len(x[0])
    uniqeLables = np.unique(y)
    numberoflabels = len(uniqeLables)
    avg = np.mean(x, axis=0)
    meanofeach = []
    # define between and within scateers :
    WithinScatter, BetweenScatter = np.zeros(
        (NumberOfFeatures, NumberOfFeatures)), np.zeros((NumberOfFeatures, NumberOfFeatures))

    # calculate formulas :
    for i, j in enumerate(uniqeLables):
        SampleOFClassY = x[y == j]
        q = np.where(y == j)
        averegaeOfEachfeature = np.mean(SampleOFClassY, axis=0)
        meanofeach.append(averegaeOfEachfeature)
        WithinScatter = WithinScatter + \
            ((SampleOFClassY - averegaeOfEachfeature).T.dot((SampleOFClassY - averegaeOfEachfeature)))

        BetweenScatter = BetweenScatter + (SampleOFClassY.shape[0] * (averegaeOfEachfeature - avg).reshape(
            NumberOfFeatures, 1).dot((averegaeOfEachfeature - avg).reshape(NumberOfFeatures, 1).T))

    return WithinScatter, BetweenScatter, NumberOfFeatures, avg, meanofeach

# %%


def GeneralSoloution(WithinMatrix, BetweinMatrix, n):
    Matrix = np.linalg.inv(WithinMatrix+0.001*np.eye(n)).dot(BetweinMatrix)
    Evalue, Evector = np.linalg.eig(Matrix)

    return Evector, Evalue


# %%
def GeneralSoloution2(WithinMatrix, BetweinMatrix):
    Matrix = np.linalg.pinv(WithinMatrix).dot(BetweinMatrix)
    Evalue, Evector = np.linalg.eig(Matrix)

    return Evector, Evalue

# %%


def sorteigens(eigenVectors, eigenValues):
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    return eigenVectors, eigenValues

# %%
# def transform(x , eigenvectors , k):
#     eigenvectors = np.array(eigenvectors)
#     T = np.dot(x , eigenvectors[:k].T)

#     b = (eigenvectors[:, :k].T @ x.T).T
#     return b

# %%
# def reconstruction(eigenvectors,Transformer,k,avg,MeanOfEach):
#     MeanOfEach = np.array(MeanOfEach)
#     T = eigenvectors[:,:k]
#     reconstruct = np.dot(Transformer ,T.T)
#     # reconstruct =  reconstruct
#     b = eigenvectors[:, :Transformer.shape[1]] @ Transformer.T

#     b = b.reshape(400,64*64)
#     # print(b.shape)
#     b = b +avg
#     return b

# %%


def transform(x, eigenvectors, k):
    eigenvectors = np.array(eigenvectors)
    T = np.dot(x, eigenvectors[:k].T)

    b = (eigenvectors[:, :k].T @ x.T).T
    return T


def reconstruction(eigenvectors, Transformer, k, avg, MeanOfEach):
    MeanOfEach = np.array(MeanOfEach)
    T = eigenvectors[:, :k]
    reconstruct = np.dot(Transformer, T.T)
    reconstruct = reconstruct+avg
    b = eigenvectors[:, :Transformer.shape[1]] @ Transformer.T

    b = b.reshape(400, 64*64)
    # print(b.shape)
    b = b + avg
    return reconstruct


# %%
WithinMatrixes, BetwenMatrix, n, avg, meanofeach = CalculateScaters(x, y)

# %%
eigenvectors, eigenvalues = GeneralSoloution(WithinMatrixes, BetwenMatrix, n)


# %%
Eigenvectors, Eigenvalues = sorteigens(eigenvectors, eigenvalues)

# %%
X1 = transform(x, Eigenvectors, 1)
X40 = transform(x, Eigenvectors, 40)
X20 = transform(x, Eigenvectors, 20)
X60 = transform(x, Eigenvectors, 60)
X80 = transform(x, Eigenvectors, 80)
X200 = transform(x, Eigenvectors, 200)

# %%
XR1 = reconstruction(Eigenvectors, X1, 1, avg, meanofeach)
XR40 = reconstruction(Eigenvectors, X40, 40, avg, meanofeach)
XR60 = reconstruction(Eigenvectors, X60, 60, avg, meanofeach)
XR20 = reconstruction(Eigenvectors, X20, 20, avg, meanofeach)
XR80 = reconstruction(Eigenvectors, X80, 80, avg, meanofeach)
XR200 = reconstruction(Eigenvectors, X200, 200, avg, meanofeach)

# %%

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
fig.axes[0].imshow(XR1[180].reshape(64, 64), cmap='gray')
fig.axes[0].set_title('k = 1')
fig.axes[1].imshow(XR20[180].reshape(64, 64), cmap='gray')
fig.axes[1].set_title('k = 20')
fig.axes[2].imshow(XR40[180].reshape(64, 64), cmap='gray')
fig.axes[2].set_title('k = 40')
fig.axes[3].imshow(XR60[180].reshape(64, 64), cmap='gray')
fig.axes[3].set_title('k = 60')
fig.axes[4].imshow(XR80[180].reshape(64, 64), cmap='gray')
fig.axes[4].set_title('k = 80')
fig.axes[5].imshow(XR200[180].reshape(64, 64), cmap='gray')
fig.axes[5].set_title('k = 200')
fig.suptitle('different K ', size=30)
fig.tight_layout()
plt.show()

# %%


fig, axes = plt.subplots(3, 4, figsize=(10, 7))

for i, ax in enumerate(fig.axes):
    ax.imshow(Eigenvectors[:, i].reshape(64, 64), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.suptitle('eignevectors reshape', size=20)
fig.tight_layout()
plt.show()

# %%
# def mean_squared_error(y_true, y_pred):
#     return np.mean(np.mean((y_true-y_pred)**2, axis=1))

# eigenvectors_num = 500
# step_size = 1
# fig, ax = plt.subplots(figsize=(8, 6))
# errors = []
# for k in range(1, eigenvectors_num, step_size):
#     a= transform(x , Eigenvectors , k)
#     d=reconstruction(Eigenvectors , a , k,avg,meanofeach)
#     errors.append(mean_squared_error(x, d))
# ax.scatter(range(1, eigenvectors_num, step_size), errors, s=6)
# ax.plot(range(1, eigenvectors_num, step_size), errors, alpha=0.5)
# ax.set_xlabel('Number of components')
# ax.set_ylabel('MSE')
# fig.suptitle('MSE between the original and reconstructed images')
# plt.tight_layout()
# plt.show()

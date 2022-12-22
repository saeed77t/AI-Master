# %%
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt

# %%


class PCA:
    dataset = []
    samples = []
    labels = []
    classes = []

    m = None
    n = None

    samples_mean = None
    sample_centered = None
    eigen_vectors = None
    eigen_values = None

    variance_explained = None
    cumulative_variance_explained = None

    transformed_samples = []
    reconstructed_samples = []

    def __init__(self, dataset_name="olivetti"):
        if(dataset_name == "olivetti"):
            self.load_olivetti()
        else:
            raise Exception("Sorry, we don't support that yet, maybe later :)")

    def load_olivetti(self):
        rng = np.random.RandomState(0)
        self.dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
        self.samples = self.dataset.images
        self.labels = self.dataset.target
        classes = np.unique(self.labels)
        classes.sort()
        self.classes = classes

    def fit(self):
        self.samples_mean = np.mean(self.samples, axis=0)
        sample_centered = self.samples - self.samples_mean
        self.m = sample_centered.shape[0]
        self.n = sample_centered.shape[2]
        self.sample_centered = sample_centered.reshape((self.m, self.n*self.n))
        cov_mat = np.cov(self.sample_centered.T)

        # eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
        eigen_vectors, self.eigen_values, vh = np.linalg.svd(cov_mat)
        self.eigen_vectors = eigen_vectors.T

        self.variance_explained = [
            (eigen_value/sum(self.eigen_values))*100 for eigen_value in self.eigen_values]
        self.cumulative_variance_explained = np.cumsum(self.variance_explained)

    def transform(self, n_pcs):
        eigenvector_subset = self.eigen_vectors[:n_pcs]
        self.transformed_samples = np.dot(
            eigenvector_subset, self.sample_centered.T).T
        return self.transformed_samples

    def reconstruct(self, n_pcs):
        eigenvector_subset = self.eigen_vectors[:n_pcs]
        reconstructed_samples = np.dot(
            self.transformed_samples, eigenvector_subset)
        reconstructed_samples = reconstructed_samples.reshape(
            self.m, self.n, self.n)
        self.reconstructed_samples = reconstructed_samples + self.samples_mean
        return self.reconstructed_samples

    def best_n_components(self, capture_rate=99):
        nth_component = np.argwhere(
            self.cumulative_variance_explained > capture_rate).flatten()[0] + 1
        return nth_component

    def plot_first_20_components(self):
        fig, axes = plt.subplots(4, 5, figsize=(8, 6))
        for i, ax in enumerate(fig.axes):
            ax.imshow(self.eigen_vectors[i].reshape(
                self.n, self.n), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.suptitle(f"20 first principal components", size=14)
        fig.tight_layout()
        plt.show()

    def plot_cumulative_variace(self, k_components):
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(range(1, k_components+1),
                    self.cumulative_variance_explained[:k_components], c='b', s=5)
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative explained variance')
        fig.suptitle("Explained variance vs Number of components")
        fig.tight_layout()
        plt.show()

    def visualize_samples(self, samples, title=None):
        fig, axes = plt.subplots(2, 5, figsize=(6, 3))
        for i, ax in enumerate(fig.axes):
            ax.imshow(samples[i], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.suptitle(title, size=14)
        fig.tight_layout()
        plt.show()

    def visualize_single_sample(self, sample, title):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(sample, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.suptitle(title, size=14)
        fig.tight_layout()
        plt.show()


# %%
pca = PCA()
pca.visualize_samples(pca.samples, "Some samples from Olitive dataset")
pca.fit()

# %%
transformed_2d = pca.transform(2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(transformed_2d[:, 0], transformed_2d[:, 1], color="purple", s=8)
plt.show()

# %%
transformed_3d = pca.transform(3)

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='3d'))
ax.scatter(transformed_3d[:, 0], transformed_3d[:, 1],
           transformed_3d[:, 2], color="purple", s=10)
plt.show()

# %%
diffrent_k = [1, 20, 50, 150]
for k in diffrent_k:
    pca.transform(k)
    pca.reconstruct(k)
    pca.visualize_samples(pca.reconstructed_samples,
                          f'Rconstruced PCA images (k = {k})')

# %%


def mean_squared_error(y_true, y_pred):
    return np.mean(np.mean((y_true-y_pred)**2, axis=1))


eigenvectors_num = 1000
step_size = 10
fig, ax = plt.subplots(figsize=(8, 6))
errors = []
for k in range(1, eigenvectors_num, step_size):
    pca.transform(k)
    pca.reconstruct(k)
    errors.append(mean_squared_error(pca.samples, pca.reconstructed_samples))
ax.scatter(range(1, eigenvectors_num, step_size), errors, s=6)
ax.plot(range(1, eigenvectors_num, step_size), errors, alpha=0.5)
ax.set_xlabel('Number of components')
ax.set_ylabel('MSE')
fig.suptitle('MSE between the original and reconstructed images')
plt.tight_layout()
plt.show()

# %%
pca.plot_first_20_components()

# %%
pca.plot_cumulative_variace(300)

# %%
print(
    f"Best number of PCs are to keep 85% of the variance: {pca.best_n_components(85)}")
print(f"{pca.best_n_components(75)} components are needed to keep 75% of the variance")
print(f"{pca.best_n_components(95)} components are needed to keep 95% of the variance")
print(
    f"{round(pca.variance_explained[0], 2)}% variance is retained by the first component")
print(
    f"{round(pca.cumulative_variance_explained[4], 2)}% variance is retained by the first 5 components")

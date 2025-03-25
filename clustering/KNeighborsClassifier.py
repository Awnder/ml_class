import numpy as np
from sklearn.utils import Bunch


class KNeighborsClassifier:
	def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
		"""Bare initialization of KNeighborsClassifier class
		Parameters:
			n_neighbors: int, number of neighbors to use by default for kneighbors queries
			metric: str, the distance metric to use for the tree, can be euclidean, manhattan, or cosine
		"""
		self.metric = metric
		self.n_neighbors = n_neighbors
		self.data = None
		self.labels = None
		self.classes_ = None
		self.n_features_in_ = None
		self.feature_names_in_ = None
		self.n_samples_fit_ = None

	def fit(self, X: np.ndarray, y: np.ndarray) -> object:
		"""Fit the model using X as training data and y as target values
		Parameters:
			X_train: np.ndarray, of shape (n_samples, n_features) or sklearn.utils.Bunch for feature names
				If Bunch, expects the following attributes:
					data_bunch = Bunch(
						data=np.array([[1, 2], [3, 4], [5, 6]]),
						target=np.array([0, 1, 0]),
						feature_names=['feature1', 'feature2'],
						target_names=['class1', 'class2']
					)
			y_train: np.ndarray, of shape (n_samples)
		Returns:
			self: object
		"""
		if isinstance(X, Bunch) and hasattr(X, "feature_names"):
			self.n_features_in_ = len(X.data[0])
			self.n_samples_fit_ = len(X.data)
			self.feature_names_in_ = X.feature_names
		else:
			self.n_features_in_ = X.shape[1]
			self.n_samples_fit_ = X.shape[0]

		X = X.data if isinstance(X, Bunch) else X

		self.data = np.array(X)
		self.labels = np.array(y)

		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict the class labels for the provided data
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
		Returns:
			y: np.ndarray, of shape (n_samples)
		"""
		neigh_ind = self.kneighbors(X, self.n_neighbors, return_distance=False)

		# count the labels and return the majority class
		# ai to help with this one
		y_pred = np.array([np.argmax(np.bincount(self.labels[indices])) for indices in neigh_ind])
		return y_pred

	def kneighbors(self, X: np.ndarray = None, n_neighbors: int = None, return_distance: bool = True) -> tuple[np.ndarray, np.ndarray]:
		"""Find the K-neighbors of a point.
		Parameters:
			X: np.ndarray, optional
				The input data to find the neighbors for. If None, the training data will be used.
			n_neighbors: int, optional
				Number of neighbors to get. If None, the default number of neighbors will be used.
			return_distance: bool, default=True
				Whether or not to return the distances.
		Returns:
			neigh_dist: np.ndarray, of shape (n_samples, n_neighbors)
				Array representing the distances to points, only present if return_distance=True.
			neigh_ind: np.ndarray, of shape (n_samples, n_neighbors)
				Indices of the nearest points in the population matrix.
		"""
		if self.data is None and X is None:
			raise ValueError("fit the model first or provide data")

		data = self.data if X is None else X
		n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

		distances = None
		if self.metric == "euclidean":
			distances = self._euclidean_distance(X, data)
		elif self.metric == "manhattan":
			distances = self._manhattan_distance(X, data)
		elif self.metric == "cosine":
			distances = self._cosine_distance(X, data)
		else:
			raise ValueError("metric must be euclidean, manhattan, or cosine")
			
		# diagonal is the distance to itself, so set to infinity so points aren't counted as their own neighbor
		np.fill_diagonal(distances, np.inf)

		neigh_dist = np.sort(distances, axis=1)[:, :self.n_neighbors] # slice to get 0 to n_neighbors
		neigh_ind = np.argsort(distances, axis=1)[:, :self.n_neighbors]

		if return_distance:
			return neigh_dist, neigh_ind
		else:
			return neigh_ind

	def score(self, X: np.ndarray, y: np.ndarray) -> float:
		"""Return the mean accuracy on the given test data and labels
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features), test samples
			y: np.ndarray, of shape (n_samples), true labels for X
		Returns:
			score: float, mean accuracy of self.predict(X) with respect to y
		"""
		# accuracy: correct predictions / total predictions
		# y_pred == y compares each prediction to see if correct, returns bool array
		# np.mean calculates avg of this array to get a score between 0 and 1
		return np.mean(self.predict(X) == y)

	def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
		"""
		p1: np.ndarray of numpy points, p2: np.ndarray of numpy points
		returns: np.ndarray of scalar distances between each point using euclidean
		"""
		return np.linalg.norm(p1[:, np.newaxis, :] - p2[np.newaxis, :, :], axis=2)

	def _manhattan_distance(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
		"""
		p1: np.ndarray of numpy points, p2: np.ndarray of numpy points
		returns: np.ndarray of scalar distances between each point using manhattan
		"""
		return np.sum(np.abs(p1[:, np.newaxis, :] - p2[np.newaxis, :, :]), axis=2)

	def _cosine_distance(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
		"""
		p1: np.ndarray of numpy points, p2: np.ndarray of numpy points
		returns: np.ndarray of scalar distances between each point using 1 - cosine similarity
		"""
		return 1 - (np.dot(p1, p2.T) / (np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1)))


if __name__ == "__main__":
	knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
	data_bunch = Bunch(data=np.array([[1, 2], [1, 1], [70, 70]]), target=np.array([0, 0, 1]), feature_names=["feature1", "feature2"], target_names=["class1", "class2"])

	knn = knn.fit(data_bunch, data_bunch.target)

	p = knn.predict(np.array([[0, 2], [1, 1], [71, 70]]))

	s = knn.score(np.array([[0, 2], [1, 1], [71, 70]]), np.array([0, 0, 1]))
	print(s)
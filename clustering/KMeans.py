import numpy as np
import random

class KMeans:
	def __init__(self, n_clusters: int=8, max_iter: int=100, random_state: int=None):
		''' 
		initialization of KMeans class 
		Parameters:
			n_clusters: int, number of clusters to form and centroids to generate
			max_iter: int, maximum number of iterations of the k-means algorithm for a single run
			random_state: int, random generation of centroids, current code is always random
		'''
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.random_state = random_state
		self.cluster_centers_ = None
		self.labels_ = None
		self.inertia_ = None
		self.n_iter_ = None

	def fit(self, X: np.ndarray, y: None=None) -> object:
		'''
		compute k-means clustering with training data to find centroids
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
			y: ingored, used for compatibility with sklearn KMeans implementation
		Returns:
			self: object
		'''
		self.cluster_centers_ = self._generate_centroids(X)
		
		for i in range(self.max_iter):
			new_labels = self._label_points(X)
			new_inertia = self._calculate_inertia(X)

			# if newly calculated inertia is less than previous iteration and the labels are different
			# remember smaller inertia is better. if the labels are the same but we've found better inertia, we might be a loop, so break
			if self.inertia_ is None or (self.inertia_ > new_inertia and not np.array_equal(self.labels_, new_labels)):
				self.labels_ = new_labels
				self.inertia_ = new_inertia
				self.n_iter_ = i
			else:
				break

		return self
	
	def predict(self, X: np.ndarray) -> np.ndarray:
		'''
		Predicts closest cluster for each sample in new data X
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
		Returns:
			labels: np.ndarray, of shape (X.size[0])
		'''
		return self._label_points(X)


	def fit_predict(self, X: np.ndarray, y: None=None) -> np.ndarray:
		''' 
		Compute k-means clustering and return labels of the inputted data
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
			y: ingored, used for compatibility with sklearn KMeans implementation
		Returns:
			self.labels_: np.ndarray, of shape (n_samples)
		'''
		self.fit(X)

		return self.labels_
	
	def score(self) -> np.float32:
		''' 
		Opposite of the inertia indicating how well the KMeans algorithm is performing
		Returns:
			-self.inertia_: np.float32 
		'''
		return -self.inertia_

	def _generate_centroids(self, X: np.ndarray) -> np.ndarray:
		''' 
		Generate n_clusters number of centroids by picking samples from the dataset, 
		if self.random_state is an integer, use it as a seed to stablize centroid creation
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
		Returns:
			centroids: np.ndarray, of shape (n_clusters, n_features)
		'''
		centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=np.int32)

		if self.random_state is not None:
			random.seed(self.random_state)
			
		past_indecies = []

		for _ in range(self.n_clusters):
			random_index = random.randint(0, len(X)-1)
			while random_index in past_indecies: 
				random_index = random.randint(0, len(X)-1)

			past_indecies.append(random_index)
			
		for i in range(len(past_indecies)):
			centroids[i] = X[past_indecies[i]]
		
		return centroids
	
	def _label_points(self, X: np.ndarray) -> np.ndarray:
		'''
		Assign each point to the closest centroid using Euclidean distance
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
		Returns:
			labels: np.ndarray, of shape (n_samples)
		'''
		labels = np.zeros(len(X), dtype=np.int32)

		for i in range(len(X)):
			point = X[i]
			distance = np.sqrt(np.sum((self.cluster_centers_ - point)**2, axis=1))
			labels[i] = np.argmin(distance) # gets the index of the minimum distance

		return labels

	def _calculate_inertia(self, X: np.ndarray) -> np.float32:
		'''
		Calculate sum of squared distances of samples to their closest cluster center
		Parameters:
			X: np.ndarray, of shape (n_samples, n_features)
		Returns:
			inertia: np.float32
		'''
		inertia = 0.0

		for i in range(len(X[0])):
			# not squaring b/c inertia is the sum of squared distances
			distance = np.sum((self.cluster_centers_ - X[i])**2, axis=1)
			inertia += np.min(distance)

		return np.sum(inertia)
	

# kmeans = KMeans(n_clusters=2)
# data = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
# kmeans.fit(data)
# print(kmeans.cluster_centers_)
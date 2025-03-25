import numpy as np

def manhattan(p1: list, p2: list) -> list:
  ''' 
  p1: list of numpy points, p2: list of numpy points
  returns: a list of scalar distances between each point using manhattan
  '''
  return np.sum(np.abs(p1[:, np.newaxis, :] - p2[np.newaxis, :, :]), axis=2)

def chebyshev(p1: list, p2: list) -> list:
  ''' 
  p1: list of numpy points, p2: list of numpy points
  returns: a list of scalar distances between each point using chebyshev
  '''
  return np.max(np.abs(p1[:, np.newaxis, :] - p2[np.newaxis, :, :]), axis=2)

def cosine_distance(p1: list, p2: list) -> list:
  ''' 
  p1: list of numpy points, p2: list of numpy points
  returns: a list of scalar distances between each point using 1 - cosine similarity
  '''
  return 1 - (np.dot(p1, p2.T) / (np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1)))

print('1st test:')

p1 = np.array([[0,1,2]])
p2 = np.array([[2,2,2]])
print(manhattan(p1, p2))
print(chebyshev(p1, p2))
print(cosine_distance(p1, p2))

print('\n2nd test:')

p1 = np.array([[0,1,2], [1,1,1]])
p2 = np.array([[2,2,2], [3,3,3]])
print(manhattan(p1, p2))
print(chebyshev(p1, p2))
print(cosine_distance(p1, p2))
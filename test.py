from main import gram_schmidt
import numpy as np

vectors = np.array([[1, 1, 1], [1, 0, -1], [1, -2, 1]])
A = np.array([[2, -1, 0], 
              [-1, 2, -1], 
              [0, -1, 2]])

orthogonal_vectors = gram_schmidt(vectors, A)
print('1', orthogonal_vectors) # Expected output: [[1, 1, 1], [1, 0, -1], [0, -3, 0]]

vectors = np.array([[1, 1], [-1, 2]])
A = np.array([[2, 1], [1, 2]])

orthogonal_vectors = gram_schmidt(vectors, A)
print('2', orthogonal_vectors) # Expected output: [[1, 1], [-1.5, 1.5]]

vectors = np.array([[1, 1], [1, 1]])
A = np.array([[2, 1], [1, 2]])
try:
    orthogonal_vectors = gram_schmidt(vectors, A)
except ValueError as e:
    print('3', e) # Expected output: Vectors are not linearly independent.

vectors = np.array([[1, 0, 1], [2, 0, -1]])
A = np.array([[1, 2, 0], 
              [2, 1, 2], 
              [0, 2, 1]])

orthogonal_vectors = gram_schmidt(vectors, A)
print('4', orthogonal_vectors) # Expected output: [[1, 0, 1], [1.5, 0, -1.5]]

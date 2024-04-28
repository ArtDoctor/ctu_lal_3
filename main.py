import numpy as np
from utils import get_number_input, get_matrix_input, determinant, get_column_input, dot, GEM


def is_positive_definite(matrix: np.ndarray) -> bool:
    """ Check if a matrix is positive definite using Sylvester's criterion. """
    for i in range(1, len(matrix) + 1):
        if determinant(matrix[:i, :i]) <= 0:
            return False
    return True


def scalar_product(u: list[np.ndarray], v: list[np.ndarray], A: np.ndarray) -> float:
    """ Compute the scalar product using matrix A. """
    return dot(u, dot(A, v))


def check_linear_independence(vectors: list[np.ndarray]) -> bool:
    """ Check if a list of vectors is linearly independent."""
    # Perform Gaussian Elimination to obtain the row echelon form of the matrix
    row_echelon = GEM(np.array(vectors))
    
    # Count the number of non-zero rows in the row echelon form
    rank = sum(any(row != 0) for row in row_echelon)
    
    # Check if the rank is equal to the number of vectors
    return rank == len(vectors)


def gram_schmidt(vectors: list[np.ndarray], A: np.ndarray) -> list[np.ndarray]:
    """ Perform the Gram-Schmidt orthogonalization process. """
    n = len(vectors)
    orthogonal_vectors = []
    
    for i in range(n):
        temp_vec = vectors[i]
        for j in range(i):
            # Compute the projection of vectors[i] onto orthogonal_vectors[j]
            proj = scalar_product(vectors[i], orthogonal_vectors[j], A) / \
                   scalar_product(orthogonal_vectors[j], orthogonal_vectors[j], A)
            # Subtract the projection from vectors[i]
            temp_vec = temp_vec - proj * np.array(orthogonal_vectors[j])
        # Check if the resulting vector is linearly independent from previous orthogonal vectors
        if not check_linear_independence([temp_vec] + orthogonal_vectors[:i]):
            raise ValueError("Vectors are not linearly independent.")
        # Add the orthogonalized vector to the list of orthogonal vectors
        orthogonal_vectors.append(temp_vec)

    return orthogonal_vectors


def main():
    # User input for matrix dimension
    print("Enter the dimension (n) of the space R^n: ")
    n = get_number_input(1, 1000)
    
    # User input for the matrix A
    print("Enter the matrix of the corresponding quadratic form in the standard basis:")
    A = get_matrix_input(n, n)

    # Check if A is symmetric and positive definite
    if not (A == A.T).all():
        raise ValueError("Matrix A is not symmetric.")
    
    if not is_positive_definite(A):
        raise ValueError("Matrix A is not positive definite.")
    
    # User input for vectors
    print("How many vectors do you want to orthogonalize?")
    k = get_number_input(1, n)
    print("Enter the vectors:")
    vectors = []
    for i in range(k):
        vec = get_column_input(n)
        vectors.append(vec)
    if not check_linear_independence(vectors):
        raise ValueError("Vectors are not linearly independent.")
    
    # Perform Gram-Schmidt orthogonalization
    try:
        orthogonal_vectors = gram_schmidt(vectors, A)
        print("Orthogonalized vectors:")
        for i, vec in enumerate(orthogonal_vectors):
            print(f"Vector {i+1}: {vec}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()

from typing import List
import numpy as np


def get_number_input(min: int, max: int) -> int:
    """Gets an integer input from the user."""
    try:
        n = int(input("n: "))
    except ValueError:
        print("Error: Please enter a valid integer.")
        return get_number_input(min, max)
    if n < min or n > max:
        print(f"Error: The number should be between {min} and {max}.")
        return get_number_input(min, max)
    else:
        return n


def get_column_input(n_elements: int) -> List[float]:
    """Gets a vector input from the user."""
    print("Enter the elements of the column/row, separated by spaces:")
    row = list(map(float, input().split()))
    if len(row) != n_elements:
        print(f"Error: The number of elements should be {n_elements}.")
        print("Please enter the row correctly.")
        return get_column_input(n_elements)
    else:
        return row


def get_matrix_input(n: int, m: int) -> np.ndarray:
    """Gets matrix input from the user.
    n: number of rows
    m: number of columns"""
    matrix = [
        get_column_input(m) for _ in range(n)
    ]
    return np.array(matrix)


def GEM(matrix: np.ndarray) -> np.ndarray:
    """Perform Gaussian elimination on a matrix."""
    n = matrix.shape[0]
    matrix = matrix.astype(float)

    for col in range(0, n-1):
        # Check if the diagonal element is 0
        if matrix[col, col] == 0:
            for row in range(col+1, n):
                # Swap the row with a non-zero element in the same column
                if matrix[row, col] != 0:
                    saved_row = matrix[col].copy()
                    matrix[col] = matrix[row]
                    matrix[row] = saved_row
                    break

        # Eliminate the elements below the diagonal in the same column
        for row in range(n-1, col, -1):
            matrix[row] = matrix[row] - matrix[col] * (matrix[row, col] / matrix[col, col])

    return matrix


def determinant(matrix: np.ndarray) -> float:
    """Calculate the determinant of a matrix using Gaussian elimination."""
    n = matrix.shape[0]
    matrix = matrix.astype(float)
    det = 1.0

    # Reduce the matrix to upper triangular form
    matrix = GEM(matrix)

    # Calculate the determinant by multiplying the diagonal elements
    for i in range(n):
        det *= matrix[i, i]

    return det


def dot(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the dot product of two vectors."""
    result = 0.0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result

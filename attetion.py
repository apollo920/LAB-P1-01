import numpy as np 

def softmax(matrix: np.ndarray):
    shifted_values = matrix - matrix.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = exponentials.sum(axis=1, keepdims=True)
    return exponentials / row_sums
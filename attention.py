import numpy as np 

def softmax(matrix: np.ndarray):
    shifted_values = matrix - matrix.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = exponentials.sum(axis=1, keepdims=True)
    return exponentials / row_sums

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d_k = K.shape[1]
    scaling_factor = np.sqrt(d_k)

    raw_scores = Q @ K.T
    scaled_scores = raw_scores / scaling_factor
    attention_weights = softmax(scaled_scores)
    attention_output = attention_weights @ V

    return attention_output, attention_weights
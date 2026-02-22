import numpy as np


def softmax(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(
            f"softmax espera uma matriz 2D, recebeu shape {matrix.shape}."
        )
    shifted_values = matrix - matrix.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = exponentials.sum(axis=1, keepdims=True)
    return exponentials / row_sums


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K e V devem ser matrizes 2D (n_seq, d_model).")

    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"A última dimensão de Q ({Q.shape[1]}) deve ser igual "
            f"à última dimensão de K ({K.shape[1]})."
        )

    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"O número de linhas de K ({K.shape[0]}) deve ser igual "
            f"ao número de linhas de V ({V.shape[0]})."
        )

    d_k = K.shape[1]
    scaling_factor = np.sqrt(d_k)

    raw_scores = Q @ K.T
    scaled_scores = raw_scores / scaling_factor
    attention_weights = softmax(scaled_scores)
    attention_output = attention_weights @ V

    return attention_output, attention_weights
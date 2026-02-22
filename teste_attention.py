import numpy as np 
import numpy.testing as npt
from attention import scaled_dot_product_attention

Q = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
]), dtype=np.float64 

K = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
]), dtype=np.float64

V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

def _compute_reference_output(Q, K, V):
    d_k = K.shape[1]
    raw_scores = Q @ K.T / np.sqrt(d_k)
    shifted = raw_scores - raw_scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return weights @ V, weights

def test_weights_sum_to_one():
    _, weights = scaled_dot_product_attention(Q, K, V)
    row_sums = weights.sum(axis=1)
    npt.assert_array_almost_equal(row_sums, np.ones_like(row_sums), decimal=6, err_msg="Attention weights do not sum to 1")
    print("test_weights_sum_to_one passed.")

def test_output_shape():
    output, _ = scaled_dot_product_attention(Q, K, V)
    expected_shape = (Q.shape[0], V.shape[1])
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    print("teste_output_shape passed.")

def test_numerical_correctness():
    output, weights = scaled_dot_product_attention(Q, K, V)
    expected_output, expected_weights = _compute_reference_output(Q, K, V)

    npt.assert_array_almost_equal(
        weights, expected_weights, decimal=7,
        err_msg="Attention weights diferem da referência manual.",
    )
    npt.assert_array_almost_equal(
        output, expected_output, decimal=7,
        err_msg="Attention output difere da referência manual.",
    )
    print("test_numerical_correctness passed.")

def main():
    print("=" * 52)
    print("  Testes — Scaled Dot-Product Attention")
    print("=" * 52)

    print("\n[Inputs]")
    print(f"  Q:\n{Q}\n")
    print(f"  K:\n{K}\n")
    print(f"  V:\n{V}\n")

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("[Outputs]")
    print(f"  Attention Weights:\n{np.round(weights, 4)}\n")
    print(f"  Attention Output:\n{np.round(output, 4)}\n")

    print("[Executando testes...]")
    passed = 0
    failed = 0
    for test_fn in [test_weights_sum_to_one, test_output_shape, test_numerical_correctness]:
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as error:
            print(f"  [FAILED] {test_fn.__name__}: {error}")
            failed += 1

    print("\n" + "=" * 52)
    print(f"  Resultado: {passed} passou(aram), {failed} falhou(aram)")
    print("=" * 52)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
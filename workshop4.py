from itertools import product
import numpy as np
import numpy.random as rng

def binary_monomial(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.uint8)
    rows = np.arange(size)
    columns = np.arange(size)
    rng.shuffle(rows)
    rng.shuffle(columns)

    for row, column in zip(rows, columns):
        matrix[row, column] = 1

    return matrix

def words_for(word_length: int, alphabet_size: int) -> np.array:
    for word in product(range(alphabet_size), repeat=word_length):
        yield np.array(word)

def syndrome(control_matrix: np.ndarray, word: np.ndarray, alphabet_size: int) -> np.ndarray:
    return (control_matrix @ word) % alphabet_size

def is_codeword(control_matrix: np.ndarray, word: np.ndarray, alphabet_size: int) -> bool:
    return not np.any(syndrome(control_matrix, word, alphabet_size))

def generate_code(control_matrix: np.ndarray, alphabet_size: int) -> np.ndarray:
    return np.array([word for word in words_for(control_matrix.shape[1], alphabet_size) if is_codeword(control_matrix, word, alphabet_size)])

def gen_code(generator_matrix: np.ndarray, alphabet_size: int) -> np.ndarray:
    m, _ = generator_matrix.shape
    codewords = []

    for word in words_for(m, alphabet_size):
        codewords.append((word @ generator_matrix) % alphabet_size)

    return np.array(codewords)

def reed_solomon(symbols: np.ndarray, k: int, alphabet_size: int) -> np.ndarray:
    assert np.all(symbols < alphabet_size)
    n = len(symbols)

    generator_matrix = np.ones((k, n), dtype=np.int32) * symbols
    generator_matrix = generator_matrix ** np.arange(k)[:, np.newaxis]

    return generator_matrix

def weight_distribution(generator_matrix: np.ndarray, alphabet_size: int) -> dict[int, int]:
    codewords = gen_code(generator_matrix, alphabet_size)
    weights = np.count_nonzero(codewords, axis=1)
    distribution = {i: 0 for i in np.arange(alphabet_size)}

    for weight in weights:
        distribution[weight] += 1
    
    return distribution

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    rng.seed(10)

    control_matrix = np.array([
        [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    ])

    monomial = binary_monomial(control_matrix.shape[1])

    print("1. Matriz monomial generada:")
    print(monomial)

    monomial_inv = np.linalg.inv(monomial.T).astype(np.uint8)
    eq_control_matrix = (control_matrix @ monomial_inv) & 1

    print("\nMatriz de control equivalente generada:")
    print(eq_control_matrix)

    #print(f"\nCódigo equivalente generado:")
    #print(generate_code(eq_control_matrix, 2))

    generator_matrix = control_matrix
    eq_generator_matrix = generator_matrix @ monomial

    print("\n2. Matriz generadora equivalente generada:")
    print(eq_generator_matrix)

    print("\nCódigo equivalente generado:")
    print(gen_code(eq_generator_matrix, 2))

    params = (6, 2, 5)
    alphabet_size = 7
    alphabet_1 = np.arange(params[0])
    alphabet_2 = np.arange(1, 1 + params[0])
    gen_matrix_1 = reed_solomon(alphabet_1, params[1], alphabet_size)
    gen_matrix_2 = reed_solomon(alphabet_2, params[1], alphabet_size)

    print("\n3. Matriz generadora para C1:")
    print(gen_matrix_1)

    print("\nMatriz generadora para C2:")
    print(gen_matrix_2)

    print("\nDistribución de pesos para C1:")
    print(weight_distribution(gen_matrix_1, alphabet_size))

    print("\nDistribución de pesos para C2:")
    print(weight_distribution(gen_matrix_2, alphabet_size))
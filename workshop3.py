import numpy as np
from itertools import product

def words_for(word_length: int, alphabet_size: int) -> np.array:
    for word in product(range(alphabet_size), repeat=word_length):
        yield np.array(word)

def generate_code(generator_matrix: np.ndarray, alphabet_size: int) -> np.ndarray:
    m, _ = generator_matrix.shape
    codewords = []

    for word in words_for(m, alphabet_size):
        codewords.append((word @ generator_matrix) % alphabet_size)

    return np.array(codewords)


def reduce(code: np.ndarray, coordinate: int) -> np.ndarray:
    code = [list(codeword) for codeword in code]
    reduced = []

    for codeword in code:
        zeroed = codeword.copy()
        zeroed[coordinate] = 0

        if zeroed in code and zeroed not in reduced:
            reduced.append([bit for i, bit in enumerate(codeword) if i != coordinate])

    return np.array(reduced)

if __name__ == "__main__":
    BINARY = 2

    generator_matrix = np.array([
        [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    ])

    print("1. Elementos del código:")

    for codeword in generate_code(generator_matrix, BINARY):
        print(codeword)

    print("\n2. Elementos del código:")

    ALPHABET_SIZE = 5

    generator_matrix = np.array([
        [1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4]
    ])

    code = generate_code(generator_matrix, ALPHABET_SIZE)

    for codeword in code:
        print(codeword)

    print("\nReducciones:")

    for coordinate in range(ALPHABET_SIZE):
        print(f"\nCoordenada {coordinate}:")

        for word in reduce(code, coordinate):
            print(word)
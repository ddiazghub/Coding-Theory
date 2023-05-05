from sys import maxsize
from typing import Dict, List, Tuple
import numpy as np

class BinaryHelpers:
    def ones(number: int) -> int:
        return bin(number).count("1")

    def bin2dec(binary: np.array) -> int:
        decimal = 0

        for i, bit in enumerate(reversed(binary)):
            decimal += (1 << i) * bit

        return decimal

    def dec2bin(decimal: int, n: int) -> List[int]:
        return [int(bit) for bit in bin(decimal)[2:].zfill(n)]

    def bin2str(binary: np.array) -> str:
        return ''.join([str(bit) for bit in binary])

    def binaries_for(n: int) -> List[int]:
        for i in range(1 << n):
            yield BinaryHelpers.dec2bin(i, n)

class BinaryCode:
    control_matrix: np.ndarray
    _shape: Tuple[int, int]
    codewords: np.ndarray
    leaders: Dict[Tuple[int], np.ndarray]

    def __init__(self, control_matrix: List[List[int]]) -> None:
        self.control_matrix = np.array(control_matrix)
        self._shape = self.control_matrix.shape
        self.codewords = self.generate_code()
        self.leaders = self.generate_classes()

    def syndrome(self, word: np.ndarray) -> Tuple[int]:
        return tuple((self.control_matrix @ word) & 1)

    def is_codeword(self, word: np.ndarray) -> bool:
        return not any(self.syndrome(word))

    def generate_code(self) -> np.ndarray:
        return np.array([word for word in BinaryHelpers.binaries_for(self._shape[1]) if self.is_codeword(word)])

    def generate_classes(self) -> Dict[Tuple[int], np.ndarray]:
        zeros = np.zeros(self._shape[1], dtype=int)

        classes = {
            self.syndrome(zeros): zeros
        }

        for shift in reversed(range(self._shape[1])):
            leader = np.array(BinaryHelpers.dec2bin(1 << shift, self._shape[1]))
            syndrome = self.syndrome(leader)

            if syndrome not in classes:
                classes[syndrome] = leader

        return classes

    def decode(self, word: np.ndarray) -> np.ndarray:
        syndrome = self.syndrome(word)
        leader = self.leaders[syndrome]
        
        return (word - leader) & 1

    def format_syndromes(self) -> str:
        syndrome_len = len(str(tuple(range(self._shape[0]))))
        header = f"| {'Líder'.center(self._shape[1])} | {'Síndrome'.center(syndrome_len)} |"
        separator = f"+{'-' * (self._shape[1] + 2)}+{'-' * (syndrome_len + 2)}+"
        body = "\n".join([f"{separator}\n| {BinaryHelpers.bin2str(leader)} | {str(syndrome).center(syndrome_len)} |" for (syndrome, leader) in self.leaders.items()])
        
        return f"{separator}\n{header}\n{body}\n{separator}"

    def min_dist(self) -> int:
        min_d = maxsize

        for i, element1 in enumerate(self.codewords):
            for element2 in self.codewords[i + 1:]:
                distance = BinaryCode.hamming(element1, element2)

                if distance < min_d:
                    min_d = distance

        return min_d

    def hamming(word1: np.array, word2: np.array) -> int:
        return np.sum(word1 != word2)


if __name__ == "__main__":
    # Se crea el código a partir de la matriz de control
    code = BinaryCode([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ])

    # Mensajes a decodificar
    messages = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ])

    print("Elementos del código:")
    print(f"{{{', '.join([BinaryHelpers.bin2str(codeword) for codeword in code.codewords])}}}")
    print()
    print("Tabla de síndromes:")
    print(code.format_syndromes())
    print()
    print("Decodificación de mensajes:")

    for message in messages:
        print(f"{BinaryHelpers.bin2str(message)} -> {BinaryHelpers.bin2str(code.decode(message))}")

    print()
    print("Distancia mínima: ", code.min_dist())
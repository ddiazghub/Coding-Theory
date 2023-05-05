"""Código fuente con funciones para el taller 1 de teoría de códigos"""
from __future__ import annotations
from typing import List, Tuple
from sys import maxsize
import math

class Code:
    codewords: List[str]
    length: int
    size: int
    min_distance: int

    def __init__(self, elements: List[str]) -> None:
        size = len(elements)
        assert size > 0
        length = len(elements[0])

        for element in elements[1:]:
            assert len(element) == length

        self.codewords = elements
        self.size = size
        self.length = length
        self.min_distance = self.get_min_distance()

    """Distancia de hamming"""
    def hamming(word1: str, word2: str) -> int:
        assert len(word1) == len(word2)
        distance = 0

        for char1, char2 in zip(word1, word2):
            if char1 != char2:
                distance += 1

        return distance

    """Distancia mínima"""
    def get_min_distance(self) -> int:
        min_dist = maxsize

        for i, element1 in enumerate(self.codewords):
            for element2 in self.codewords[i + 1:]:
                distance = Code.hamming(element1, element2)

                if distance < min_dist:
                    min_dist = distance

        return min_dist

    """Parámetros"""
    def params(self) -> Tuple[int, int, int]:
        return (self.length, self.size, self.min_distance)

    """Decodificación"""
    def decode(self, word: str) -> str:
        assert len(word) == self.length
        min_dist = Code.hamming(self.codewords[0], word)
        codeword = self.codewords[0]

        for cw in self.codewords[1:]:
            distance = Code.hamming(cw, word)

            if distance < min_dist:
                min_dist = distance
                codeword = cw

        return codeword

    """Generar todas las cadenas binarias de longitud n"""
    def generate_binary_strings(n: int) -> List[str]:
        return [bin(i)[2:].zfill(n) for i in range(2**n - 1)]


"""Solución del taller"""
if __name__ == "__main__":
    code = Code([
        "1000010001011011000110010",
        "1100010001000010001010011",
        "0000101101110111111000100",
        "1111010101110110000100111",
        "1111011011000110010010011",
        "1000001000111010111001001",
        "1111111001010010101011111",
        "0011111111111010010010111",
        "0010010111010010000100101",
        "1101110111010101001000101",
        "0010000100010001110111110",
        "0110101101000100011100111",
        "1010100000011110011001110",
        "0001111110111100101010001",
        "1010000101010101101111011",
        "1011100100010111110110111",
        "1110110101110011110100101",
        "0011011011100110011101001",
        "0101110100011001011111001",
        "1010010101101011001000001"
    ])

    print(f"Código = {{ {', '.join(code.codewords)} }}\n")

    """1. Parámetros del código."""
    n, m, d = code.params()
    print(f"1. Parámetros: ", (n, m, d))

    """2. Número de errores que puede detectar."""
    print(f"\n2. # de errores que puede detectar: ", d - 1)

    """3. Número de errores que puede corregir."""
    print(f"\n3. # de errores que puede corregir: ", (d - 1) / 2)

    """4. Decodifique los siguientes mensajes recibidos:"""
    messages = [
        "1111010001011011000110010",
        "1111110001000010001010011",
        "1110101101110111111000100",
        "1111001000111010111001001",
        "1011011011100100010101001"
    ]

    print("\n4. Mensajes a decodificar:", messages)
    print()

    for message in messages:
        codeword = code.decode(message)
        print(f"{message} -> {codeword}. Distancia: {Code.hamming(message, codeword)}")

    """5. ¿Cuántos elementos tiene la bola B_9(1110110101110011110100101)? ¿Existe otro codeword en esa bola? En caso afirmativo, muéstrelo."""
    center = "1110110101110011110100101"
    radius = 9
    
    n_elements = sum([math.comb(n, i) for i in range(radius + 1)])
    print("\nNúmero de elementos en la bola:", n_elements)

    codewords_in_sphere = [codeword for codeword in code.codewords if Code.hamming(codeword, center) <= radius]

    if len(codewords_in_sphere) > 1:
        print("Existen otros codewords en la bola:", codewords_in_sphere)
    else:
        print("No hay codewords adicionales en la bola")
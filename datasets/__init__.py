import os.path
from typing import Tuple, List, Any

import numpy as np
from nltk import wordpunct_tokenize
from numpy import ndarray, dtype, generic

from config import ROOT_DIR


def read_file(path: str) -> str:
    """
    Read text file from the file path
    :param path: path of the file
    :return: the file content
    """
    with open(path, 'r') as file:
        result = file.read()
    return result


def tokenize(content: str) -> list[str]:
    """
    Tokenize a string into token's
    :param content: content to tokenize
    :return: list of tokens
    """
    result = []

    for line in content.split("\n"):
        # Normalize the line
        line = line.lower()
        result += wordpunct_tokenize(line)
        result.append("\n")

    return result


def create_mapping(tokens: list[str]) -> tuple[dict[str:int], dict[int:str]]:
    """
    Create mapping (word -> index) and inverse mapping (index -> word) for tokens
    :param tokens: tokens
    :return: mapping and inverse mapping
    """
    vocab = set(tokens)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    return word_to_idx, idx_to_word


def split_into_features_and_labels(tokens: list[int], sequence_length=25) -> tuple[
    ndarray[Any, dtype[generic | generic | Any]], ndarray[Any, dtype[generic | generic | Any]]]:
    """
    Convert tokens into a training dataset based on the length
    :param tokens: tokens
    :param sequence_length: length of features and labels
    :return: features and labels
    """
    x = []
    y = []

    sequences = np.split(tokens, np.arange(sequence_length + 1, len(tokens), sequence_length + 1))

    # Remove the final element if the line is not long enough
    if len(sequences[-1]) < sequence_length + 1:
        sequences = sequences[:-1]

    for sequence in sequences:
        x.append(sequence[:-1])
        y.append(sequence[1:])

    return np.array(x, dtype=int), np.array(y, dtype=int)


def convert_to_idx(tokens: list[str], word_to_idx_mapping: dict[str:int]) -> list[int]:
    """
    Convert word tokens to index
    :param tokens: tokens
    :param word_to_idx_mapping: mapping from word to index
    :return: index
    """
    return [word_to_idx_mapping[token] for token in tokens]


def convert_to_words(idxs, idx_to_word_mapping: dict[int: str]) -> list[str]:
    """
    Convert index to words
    :param idxs: index to convert
    :param idx_to_word_mapping: mapping from index to word
    :return: word
    """
    return [idx_to_word_mapping[idx.item()] for idx in idxs]


if __name__ == "__main__":
    file_path = os.path.join(ROOT_DIR, "rawdata", "truyen_kieu.txt")
    content = read_file(file_path)
    tokens = tokenize(content)
    print(tokens)

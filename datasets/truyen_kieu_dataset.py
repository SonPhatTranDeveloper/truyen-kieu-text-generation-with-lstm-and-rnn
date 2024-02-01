import os.path
from typing import Any

import torch
from numpy import ndarray, dtype, generic
from torch.utils.data import Dataset, DataLoader
from datasets import (read_file, tokenize, create_mapping,
                      convert_to_idx, split_into_features_and_labels)
from config import ROOT_DIR


class TextGenerationDataset(Dataset):
    def __init__(self, file_path, sequence_length, device="mps"):
        """
        Initialize an instance of text generation dataset
        :param file_path: the path containing the text
        :param sequence_length: the length of a sequence
        """
        # Set properties
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.x, self.y, self.word_to_idx, self.idx_to_word = self._read_dataset()
        # Convert to Tensor
        self.x = torch.from_numpy(self.x).to(device)
        self.y = torch.from_numpy(self.y).to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _read_dataset(self) -> tuple[
        ndarray[Any, dtype[generic | generic | Any]], ndarray[Any, dtype[generic | generic | Any]], Any, Any]:
        """
        Read the dataset from the file path
        :return: X (the input features), Y (the labels)
        """
        file_content = read_file(self.file_path)
        tokens = tokenize(file_content)
        word_to_idx, idx_to_word = create_mapping(tokens)
        index = convert_to_idx(tokens, word_to_idx)
        x, y = split_into_features_and_labels(index, self.sequence_length)
        return x, y, word_to_idx, idx_to_word


if __name__ == "__main__":
    # Define the file path and test run
    text_file_path = os.path.join(ROOT_DIR, "rawdata", "truyen_kieu.txt")

    # Create dataset
    dataset = TextGenerationDataset(text_file_path, 25, "mps")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Display the features and labels
    features, labels = next(iter(dataloader))
    print(features.shape, features.device)
    print(labels.shape, labels.device)

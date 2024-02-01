"""
Define the GRU model used for text generation
Author: Son Phat Tran
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ROOT_DIR
from datasets.truyen_kieu_dataset import TextGenerationDataset


class GRUTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, drop_out, device):
        """
        Create a structure for GRU neural network
        """
        super().__init__()

        # Cache the parameters
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.device = device

        # Create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            device=device
        )

        # Create GRU layer
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            device=device
        )

        # Additional drop out layer
        self.dropout = nn.Dropout(
            p=drop_out
        )

        # Create linear layer
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
            device=device
        )

    def forward(self, x):
        """
        Perform the forward operation
        :param x: the input of size (batch_size, sequence_length)
        :return: prediction of size (batch_size, vocab_size)
        """

        # Pass input through embeddings to get embeddings (batch_size, sequence_length, embedding_size)
        embeddings = self.embedding(x)

        # Pass input through gru to get hidden (batch_size, sequence_length, hidden_size)
        hidden, _ = self.lstm(embeddings)

        # Pass hidden through dropout to get (batch_size, sequence_length, hidden_size)
        hidden = self.dropout(hidden)

        # Pass through linear layer to get (batch_size, sequence_length, hidden_size)
        output = self.linear(hidden)

        return output


if __name__ == "__main__":
    # Define the file path and test run
    text_file_path = os.path.join(ROOT_DIR, "rawdata", "truyen_kieu.txt")

    # Create dataset
    dataset = TextGenerationDataset(text_file_path, 25, "mps")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test the model
    model = GRUTextGenerationModel(
        vocab_size=len(dataset.word_to_idx),
        embedding_size=128,
        hidden_size=128,
        drop_out=0.3,
        device="mps"
    )

    # Pass first features and labels
    features, labels = next(iter(dataloader))

    # Get the output
    with torch.no_grad():
        output = model(features)
        print(output.shape)


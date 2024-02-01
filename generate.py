"""
Generate sequences using trained model
"""
import os

import torch
from torch.utils.data import DataLoader

from config import ROOT_DIR
from datasets.truyen_kieu_dataset import TextGenerationDataset
from datasets import convert_to_words
from models.GRUTextGeneration import GRUTextGenerationModel

import numpy as np

import time


if __name__ == "__main__":
    # Define the parameters
    # DATA_FILE_PATH = os.path.join(ROOT_DIR, "rawdata", "harry_porter_short.txt")
    # MODEL_FILE_PATH = os.path.join(ROOT_DIR, "trained", "best_harry_porter_model.pth")
    DATA_FILE_PATH = os.path.join(ROOT_DIR, "rawdata", "truyen_kieu.txt")
    MODEL_FILE_PATH = os.path.join(ROOT_DIR, "trained", "best_truyen_kieu_model.pth")
    SEQUENCE_LENGTH = 28
    DEVICE = "mps"

    # Model size
    BATCH_SIZE = 128
    EMBEDDING_SIZE = 1024
    HIDDEN_SIZE = 1024
    DROP_OUT = 0.5
    EPOCHS = 50

    # Generation parameters
    GENERATION_LENGTH = 500

    # Create dataset and data loader
    dataset = TextGenerationDataset(DATA_FILE_PATH, SEQUENCE_LENGTH, DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # Not random

    # Create model
    model = GRUTextGenerationModel(
        vocab_size=len(dataset.word_to_idx),
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        drop_out=DROP_OUT,
        device=DEVICE
    )

    # Reload the model
    best_model, word_to_idx, idx_to_word = torch.load(MODEL_FILE_PATH)
    model.load_state_dict(best_model)

    # Put model into evaluation model
    model.eval()

    # Get the first sentence
    first_sequence_idx, _ = dataset[0]
    first_sequence = convert_to_words(first_sequence_idx, dataset.idx_to_word)

    # Append the batch dimension into first_sequence_idx
    # to get (batch_size, sequence_length)
    first_sequence_idx = first_sequence_idx.reshape(1, -1)

    # Print the first sequence
    for word in first_sequence:
        if word == "\n":
            print()
        else:
            print(word + " ", end="")

    # Generate text
    with torch.no_grad():
        for i in range(GENERATION_LENGTH):
            # Pass through model to get prediction
            # y_pred has size (batch_size, sequence_length, vocab_size)
            predictions = model(first_sequence_idx)

            # Get the next vocab
            y_next = predictions[0, -1]
            y_next = torch.exp(y_next) / torch.sum(torch.exp(y_next))

            # Convert to numpy
            y_next = y_next.to('cpu').numpy()

            # Random sample vs arg max sample
            next_token = np.random.choice(y_next.shape[0], 1, p=y_next)[0]
            # next_token = y_next.argmax()

            # Append to result
            first_sequence.append(idx_to_word[next_token])

            # Update the array
            first_sequence_idx = first_sequence_idx.to('cpu').numpy()
            first_sequence_idx = np.append(first_sequence_idx, [next_token])
            first_sequence_idx = first_sequence_idx[1:].reshape(1, -1)
            first_sequence_idx = torch.from_numpy(first_sequence_idx).to(DEVICE)

            # Print the token and stop for 0.5 seconds
            if idx_to_word[next_token] == "\n":
                print("")
            else:
                print(idx_to_word[next_token] + " ", end="")
            time.sleep(0.2)

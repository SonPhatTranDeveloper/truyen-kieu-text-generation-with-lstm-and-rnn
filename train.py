"""
Train the text generation model
Author: Son Phat Tran
"""
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from config import ROOT_DIR
from datasets.truyen_kieu_dataset import TextGenerationDataset
from models.GRUTextGeneration import GRUTextGenerationModel


if __name__ == "__main__":
    # Define the parameters
    DATA_FILE_PATH = os.path.join(ROOT_DIR, "rawdata", "harry_porter_short.txt")
    MODEL_FILE_PATH = os.path.join(ROOT_DIR, "trained", "best_harry_porter_model.pth")
    SEQUENCE_LENGTH = 28
    DEVICE = "mps"

    # Model size
    BATCH_SIZE = 128
    EMBEDDING_SIZE = 1024
    HIDDEN_SIZE = 1024
    DROP_OUT = 0.5
    EPOCHS = 50

    # Create dataset and data loader
    dataset = TextGenerationDataset(DATA_FILE_PATH, SEQUENCE_LENGTH, DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model
    model = GRUTextGenerationModel(
        vocab_size=len(dataset.word_to_idx),
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        drop_out=DROP_OUT,
        device=DEVICE
    )

    # Create optimizer and cost function
    optimizer = optim.Adam(lr=1e-2, params=model.parameters())
    cost_func = nn.CrossEntropyLoss(reduction="mean")

    # Cache the best model and best cost
    best_model = None
    best_cost = float('inf')
    costs = []

    # Initialize the training loop
    for epoch in range(EPOCHS):
        # Put model in training mode
        model.train()

        # Go through training examples
        for batch_x, batch_y in dataloader:
            # Get the prediction
            y_pred = model(batch_x)

            # Flatten from (batch_size, sequence_length, C) to (batch_size * sequence_length, C)
            y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)

            # Flatten from (batch_size, sequence_length) to (batch_size * sequence_length,)
            batch_y = batch_y.reshape(-1)

            # Calculate the cost
            loss = cost_func(y_pred, batch_y)

            # Optimize the cost
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on training set cost
        model.eval()
        loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                y_pred = model(batch_x)

                # Flatten from (batch_size, sequence_length, C) to (batch_size * sequence_length, C)
                y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)

                # Flatten from (batch_size, sequence_length) to (batch_size * sequence_length,)
                batch_y = batch_y.reshape(-1)

                # Add to the cost
                loss += cost_func(y_pred, batch_y)

            # Append cost for visualization
            costs.append(loss)

            # Update the best model and best cost
            if loss < best_cost:
                best_loss = loss
                best_model = model.state_dict()

            print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

    # Save the best model
    torch.save([best_model, dataset.word_to_idx, dataset.idx_to_word], MODEL_FILE_PATH)


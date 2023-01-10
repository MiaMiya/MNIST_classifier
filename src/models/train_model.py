# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from dotenv import find_dotenv, load_dotenv
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    model.train()

    batch_size = 128
    # load the training data
    train = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()  # define type of loss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # define optimizer

    epochs = 5

    train_losses = []
    train_accuracy = []

    for e in range(epochs):
        accuracy = 0
        with tqdm(trainloader, unit="batch") as tepoch:
            for images, labels in trainloader:
                tepoch.set_description(f"Epoch {e}")

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                ps = torch.exp(outputs)
                loss = criterion(ps, labels)  # train loss
                train_losses.append(loss.item())

                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)) / len(
                    trainloader
                )

                train_accuracy.append(accuracy)

                # Backpropogate
                loss.backward()

                # Update parameters
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item() * 100)

    torch.save(model.state_dict(), "models/my_trained_model.pt")

    _, axis = plt.subplots(2)

    # For loss
    axis[0].plot(train_losses, label="loss")
    axis[0].set_title("Training loss")
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("loss")

    # For accuracy
    axis[1].plot(train_accuracy, label="accuracy")
    axis[1].set_title("Training accuracy")
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("loss")

    plt.savefig(f"reports/figures/training_curve.png")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()

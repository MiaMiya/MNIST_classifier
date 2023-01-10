# -*- coding: utf-8 -*-
import click
import numpy as np
import torch

from src.models.model import MyAwesomeModel


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_to_predict")
def evaluate(model_checkpoint, data_to_predict):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    model.eval()

    # load data for prediction
    images = np.load(data_to_predict)
    images = torch.tensor(images, dtype=torch.float)

    # Predict with our model
    outputs = model(images)
    prediction = outputs.argmax(dim=-1)
    probs = outputs.softmax(dim=-1)

    print("Predictions")
    for i in range(images.shape[0]):
        print(
            f"Image {i+1} predicted to be class {prediction[i].item()} with probability {probs[i, prediction[i]].item()}"
        )


if __name__ == "__main__":
    evaluate()

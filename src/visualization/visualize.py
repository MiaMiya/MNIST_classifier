import click
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel

@click.command()
@click.argument("model_checkpoint")
def tsne_embedding_plot(model_checkpoint):
    print(model_checkpoint)

    train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    print("Extract embeddings")
    embeddings, predicted_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            # Extract features from the backbone
            emb = model.layer1(images).reshape(images.shape[0], -1)
            embeddings.append(emb)
            predicted_labels.append(labels)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    predicted_labels = torch.cat(predicted_labels, dim=0).numpy()

    print("Running tsne")
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    for i in np.unique(predicted_labels):
        plt.scatter(embeddings_2d[predicted_labels == i, 0], embeddings_2d[predicted_labels == i, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/2d_tsne_embedding.png")


if __name__ == "__main__":
    tsne_embedding_plot()
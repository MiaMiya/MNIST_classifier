
from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
import torch
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/train_processed.pt'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.pt'), reason="Data files not found")
@pytest.mark.parametrize("file,expected_dim", [(0, torch.Size([1,28,28])), (1, torch.Size([]))])

def test_data(file, expected_dim):
    dataset_train = torch.load(f'{_PATH_DATA}/processed/train_processed.pt')
    dataset_test =torch.load(f'{_PATH_DATA}/processed/test_processed.pt')

    assert len(dataset_train[0]) == 25000, "Dataset for training does not have correct dimention"
    assert len(dataset_test[0]) == 5000, "Dataset for test does not have correct dimention"
    for images in dataset_train[file]:
        assert images.shape == expected_dim, "Shape of of input does not fit"
    assert len(dataset_train[0]) == len(dataset_train[1]), "Dataset have uneven number of images and labels"



if __name__ == "__main__":
    test_data()

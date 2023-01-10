
from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel
import torch
import pytest

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    #with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
    with pytest.raises((ValueError, RuntimeError)):
        model(torch.randn(1,2,3))

def test_model():
   model = MyAwesomeModel()
   
   assert model(torch.rand(1,1,28,28)).shape == torch.Size([1,10]), "Model output dimention not [1, 10]"


if __name__ == "__main__":
    test_model()
    test_error_on_wrong_shape()


from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel
import torch

def test_training():
   model = MyAwesomeModel()
   
   assert model(torch.rand(1,1,28,28)).shape == torch.Size([1,10]), "Model output dimention not [1, 10]"


if __name__ == "__main__":
    test_training()

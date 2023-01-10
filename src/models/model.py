from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.layer1(x))

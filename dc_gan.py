class Generator(nn.Module):
  def __init__(self):
      super().__init__()
      self.model = nn.Sequential(
      nn.Linear(100, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 784),
      nn.Tanh()
      )
  def forward(self, x):
    return self.model(x)
class Discriminator(nn.Module):
  def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
        )
  def forward(self, x):
     return self.model(x)

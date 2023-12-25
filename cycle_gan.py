class Discriminator(nn.Module):
  def __init__(self, emb_size=32):
    super(Discriminator, self).__init__()
    self.emb_size = 32

    self.label_embeddings = nn.Embedding(2, self.emb_size)
    self.model = nn.Sequential(
    nn.Conv2d(3,64,4,2,1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),
    nn.Conv2d(64,64*2,4,2,1,bias=False),
    nn.BatchNorm2d(64*2),
    nn.LeakyReLU(0.2,inplace=True),
    nn.Conv2d(64*2,64*4,4,2,1,bias=False),
    nn.BatchNorm2d(64*4),
    nn.LeakyReLU(0.2,inplace=True),
    nn.Conv2d(64*4,64*8,4,2,1,bias=False),
    nn.BatchNorm2d(64*8),
    nn.LeakyReLU(0.2,inplace=True),
    nn.Conv2d(64*8,64,4,2,1,bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2,inplace=True),
    nn.Flatten()
      )
    self.model2 = nn.Sequential(
    nn.Linear(288,100),
    nn.LeakyReLU(0.2,inplace=True),
    nn.Linear(100,1),
    nn.Sigmoid()
    )
    self.apply(weights_init)


  def forward(self,inputs,label):
    out=self.model(inputs)
    embed_label=self.label_embeddings(label)
    cat_out=torch.cat([out,embed_label],dim=1)
    model_2out=self.model2(cat_out)
    return model_2out
class Generator(nn.Module):
  def __init__(self, emb_size=32):
    super(Generator,self).__init__()
    self.emb_size = emb_size
    self.label_embeddings = nn.Embedding(2, self.emb_size)
    self.model = nn.Sequential(
    nn.ConvTranspose2d(100+self.emb_size,64*8,4,1,0,bias=False),
    nn.BatchNorm2d(64*8),
    nn.ReLU(True),
    nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
    nn.BatchNorm2d(64*4),
    nn.ReLU(True),
    nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
    nn.BatchNorm2d(64*2),
    nn.ReLU(True),
    nn.ConvTranspose2d(64*2,64,4,2,1,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64,3,4,2,1,bias=False),
    nn.Tanh()
    )
    self.apply(weights_init)
  def forward(self,input_noise,labels):
    label_embeddings = self.label_embeddings(labels) \
    .view(len(labels), \
    self.emb_size,1, 1)
    input = torch.cat([input_noise, label_embeddings], 1)
    return self.model(input)

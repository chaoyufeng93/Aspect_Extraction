class MyHingeLoss(torch.nn.Module):
  def __init__(self):
    super(MyHingeLoss, self).__init__()
    self.max_z = torch.nn.ReLU()
 
  def forward(self, asp, emb, y):
    out = -1 * self.max_z(1 - (asp*(y*emb)).sum(axis = -1))
    return out

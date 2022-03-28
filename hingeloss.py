class MyHingeLoss(torch.nn.Module):
  def __init__(self):
    super(MyHingeLoss, self).__init__()
    self.max_z = torch.nn.ReLU()
 
  def forward(self, asp, emb, y, reg_turn):

    y = y.reshape(-1,1).repeat(1,emb.shape[-1])
    out = self.max_z(1 - (asp*(y*emb)).sum(axis = -1)) + reg_turn

    return out

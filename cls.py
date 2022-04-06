class Att_Classify(torch.nn.Module):
  # weight_wv = TEXT.vocab.vectors; weight_tsm = mod.att_based_enc.trans_matrix; weight_wtm = mod.auto_enc.w_trans_matrix
  def __init__(self, vocab_size, emb_dim, k_aspect, weight_wv, weight_tsm, weight_wtm):
    super(Att_Classify, self).__init__()
    self.word_emb = torch.nn.Embedding.from_pretrained(weight_wv, freeze = True)
    self.trans_matrix = torch.nn.Linear(emb_dim, emb_dim, bias = False)
    self.w_trans_matrix = torch.nn.Linear(emb_dim, k_aspect)

    self.softmax = torch.nn.Softmax(dim = -1)

    self.trans_matrix.weight = weight_tsm.weight
    self.w_trans_matrix.weight = weight_wtm.weight
    self.w_trans_matrix.bias = weight_wtm.bias
  
  def forward(self, x):
    emb = self.word_emb(x)
    len = (x != 1).sum(dim = 1)
    y = emb.sum(dim = 1)/ len.reshape(-1,1).repeat(1, emb.shape[-1])
    y = y.reshape(x.shape[0],1,-1)
    d = torch.bmm(self.trans_matrix(emb), y.permute(0,2,1))
    att_w = self.softmax(d.permute(0,2,1)).reshape(x.shape[0],-1,1)
    out = (att_w * emb).sum(dim = 1)
    out = self.w_trans_matrix(out)
    p = self.softmax(out)
    return p

import torch
import random
import math
import numpy as np
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F

class Att_based_enc(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, weight):
    super(Att_based_enc, self).__init__()
    #fix the word embedding matrix
    #self.word_emb = torch.nn.Embedding(vocab_size, emb_dim)
    self.word_emb = torch.nn.Embedding.from_pretrained(weight, freeze = True)
    self.trans_matrix = torch.nn.Linear(emb_dim, emb_dim, bias = False)
    self.softmax = torch.nn.Softmax(dim = -1)

  def forward(self, x, ngt): # emb: batch_size, seq_len, emb_dim
    emb = self.word_emb(x) 
    emb_n = self.word_emb(ngt)
    # seq_len had been fixed.
    len = (x != 1).sum(dim = 1)
    len_n = (ngt != 1).sum(dim = 1)

    y = emb.sum(dim = 1) / len.reshape(-1,1).repeat(1,emb.shape[-1]) # batch_size, emb_dim
    y_n = emb_n.sum(dim = 1)/ len_n.reshape(-1,1).repeat(1,emb.shape[-1])

    y = y.reshape(x.shape[0],1,-1)
    d = torch.bmm(self.trans_matrix(emb), y.permute(0,2,1)) 
    #emb - row; y - columns
    att_w = self.softmax(d.permute(0,2,1)).reshape(x.shape[0],-1,1)
    out = (att_w * emb).sum(dim = 1)
    return out, y_n
  
  
  
class AutoEnc(torch.nn.Module):
  def __init__(self, k_aspect, emb_dim):
    super(AutoEnc, self).__init__()
    self.w_trans_matrix = torch.nn.Linear(emb_dim, k_aspect)
    self.softmax = torch.nn.Softmax(dim = -1)
    self.t_trans_matrix = torch.nn.Linear(k_aspect, emb_dim, bias = False)
  
  def forward(self, x):
    out = self.w_trans_matrix(x) # b_s, emb_dim, k_aspect
    p = self.softmax(out)
    out = self.t_trans_matrix(p)
    return out, self.t_trans_matrix.weight.permute(1,0),p
  
  
class Aspect_Extrac(torch.nn.Module):
  def __init__(self, word_in, emb_dim, k_aspect, weight, lambda_):
    super(Aspect_Extrac, self).__init__()
    self.k_aspect = k_aspect
    self.lambda_ = lambda_
    self.att_based_enc = Att_based_enc(word_in, emb_dim, weight)
    self.auto_enc = AutoEnc(k_aspect, emb_dim)
  
  def forward(self, x, ngt):
    emb_v, y_n = self.att_based_enc(x, ngt)
    out, t_matrix, p = self.auto_enc(emb_v)
    t_n = torch.nn.functional.normalize(t_matrix)
    reg_turn = self.lambda_*torch.norm(torch.mm(t_n,t_n.permute(1,0))- torch.eye(self.k_aspect, self.k_aspect).cuda())
    return out, y_n, reg_turn, p

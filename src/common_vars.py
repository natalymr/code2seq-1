import torch

PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
UNK_TOKEN = '<UNK>'
PAD = 0
BOS = 1
EOS = 2
UNK = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
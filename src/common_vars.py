import torch

PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
UNK_TOKEN = '<UNK>'
PAD = 0
BOS = 1
EOS = 2
UNK = 3

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
pad_token = tokenizer(tokenizer.eos_token)['input_ids'][0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

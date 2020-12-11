from dataclasses import dataclass, asdict
from typing import List, Dict

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


@dataclass
class DumpedResults:
    commit_or_file_number: str
    prefixes: Dict[str, List[float]]

    @staticmethod
    def from_model_output(file_number: str, vocabulary_probs) -> 'DumpedResults':
        pred = vocabulary_probs.max(dim=-1)[1].data.cpu().numpy().T.tolist()[0]
        predicted_target = tokenizer.decode(pred).split()
        prefixes: Dict[str, List[float]] = {}
        for ind, t in enumerate(predicted_target):
            prefixes[t] = vocabulary_probs[ind][0].tolist()
        return DumpedResults(file_number, prefixes)

    @staticmethod
    def to_json(self):
        return asdict(self)

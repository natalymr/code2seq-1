from typing import List

import torch
from nltk import bleu_score
from commit2seq.code2seq.src.common_vars import pad_token


PAD = 0
BOS = 1
EOS = 2
UNK = 3
pad_token_value = '<|endoftext|>'

special_tokens = [PAD, BOS, EOS, UNK]


class Vocab(object):
    def __init__(self, word2id={}):
        
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}    
        
    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for word in sentences:
            word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word 


def sentence_to_ids(vocab, sentence) -> List[int]:
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    ids += [EOS]
    return ids


def ids_to_sentence(vocab, ids):
    return [vocab.id2word[_id] for _id in ids]


def trim_eos(ids):
    if EOS in ids:
        return ids[:ids.index(EOS)]
    else:
        return ids


def get_nltk_bleu_score_for_corpora(refs: List[List[str]], preds: List[List[str]]) -> float:
    import nltk
    total_bleu = 0.
    for ref, pred in zip(refs, preds):
        if len(pred) == 0:
            continue
        total_bleu += nltk.translate.bleu_score.sentence_bleu([ref], pred,
                                                              auto_reweigh=True,
                                                              smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7) * 100
    return total_bleu / len(refs)



def calculate_results_set_aurora(refs, preds, target_vocab=None):
    #calc precision, recall and F1
    #same as https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L239
    print('utils')
    filterd_preds = [pred[:pred.index(EOS)] if EOS in pred else pred for pred in preds]

    filterd_preds = [list(set(pred)) for pred in filterd_preds]
    if target_vocab is not None:
        preds_in_words = [[target_vocab.id2word[p] for p in pred if p not in special_tokens] for pred in preds]
        print(f'BLEU = {bleu_score}')
        with open('top1000_results_val.txt', 'w') as results_f:
            for ref, pred in zip(refs, preds_in_words):
                results_f.write(f'{ref}\t{" ".join(pred)}\n')
        print(f'Target = {refs[:10]}')
        print(f'Predicted = {preds_in_words[:10]}')


def calculate_results_set_new_tokenizer(refs, preds, target_vocab=None):
    #calc precision, recall and F1
    #same as https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L239

    filterd_refs = refs#[ref[:ref.index(tokenizer.eos_token)] for ref in refs]
    filterd_preds = preds#[pred[:pred.index(tokenizer.eos_token)] if EOS in pred else pred for pred in preds]

#    filterd_refs = [list(set(ref)) for ref in filterd_refs]
#    filterd_preds = [list(set(pred)) for pred in filterd_preds]
    if target_vocab is not None:

        #refs_in_words = [" ".join(target_vocab.convert_ids_to_tokens(ref)).replace('\u0120', '').split() for ref in refs]
        #preds_in_words = [" ".join(target_vocab.convert_ids_to_tokens(pred)).replace('\u0120', '').split() for pred in preds]
#        print(preds)
        #refs_in_words  = [target_vocab.convert_tokens_to_string(target_vocab.convert_ids_to_tokens(ref)).replace(pad_token_value, '').split() for ref in refs]
        #preds_in_words = [target_vocab.convert_tokens_to_string(target_vocab.convert_ids_to_tokens(pred)).replace(pad_token_value, '').split() for pred in preds]
        refs_in_words = [target_vocab.decode(ref).split() for ref in refs]
        preds_in_words = [target_vocab.decode(pred).split() for pred in preds]

        #refs_in_words = [[target_vocab.convert_ids_to_tokens(r).encode('utf-8') for r in ref if r not in special_tokens] for ref in refs]
        #preds_in_words = [[target_vocab.convert_ids_to_tokens(p).encode('utf-8') for p in pred if p not in special_tokens] for pred in preds]
        bleu_score = get_nltk_bleu_score_for_corpora(refs_in_words, preds_in_words)
        print(f'BLEU = {bleu_score}')
        with open('new_tokenizer_results.txt', 'w') as results_f:
            for ref, pred in zip(refs_in_words, preds_in_words):
                results_f.write(f'{" ".join(ref).replace("<|endoftext|>", "")}\n{" ".join(pred).replace("<|endoftext|>", "")}\n\n')
        #print(f'Target = {refs_in_words[:8]}')
        #print(f'Predicted = {preds_in_words[:8]}')

    true_positive, false_positive, false_negative = 0, 0, 0

    for filterd_pred, filterd_ref in zip(filterd_preds, filterd_refs):

        for fp in filterd_pred:
            if fp in filterd_ref:
                true_positive += 1
            else:
                false_positive += 1

        for fr in filterd_ref:
            if not fr in filterd_pred:
                false_negative += 1

    # https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L282
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1, bleu_score



def calculate_results_set(refs, preds, target_vocab=None):
    #calc precision, recall and F1
    #same as https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L239
    
    filterd_refs = [ref[:ref.index(EOS)] for ref in refs]
    filterd_preds = [pred[:pred.index(EOS)] if EOS in pred else pred for pred in preds]
    
    filterd_refs = [list(set(ref)) for ref in filterd_refs]
    filterd_preds = [list(set(pred)) for pred in filterd_preds]
    if target_vocab is not None:
                
        refs_in_words = [[target_vocab.id2word[r] for r in ref if r not in special_tokens] for ref in refs]
        preds_in_words = [[target_vocab.id2word[p] for p in pred if p not in special_tokens] for pred in preds]
        bleu_score = get_nltk_bleu_score_for_corpora(refs_in_words, preds_in_words)
        print(f'BLEU = {bleu_score}')
#        with open('aurora_results.txt', 'w') as results_f:
#            for ref, pred in zip(refs_in_words, preds_in_words):
#                results_f.write(f'{ref}\t{pred}\n')
        print(f'Target = {refs_in_words[:10]}')
        print(f'Predicted = {preds_in_words[:10]}')
    
    true_positive, false_positive, false_negative = 0, 0, 0

    for filterd_pred, filterd_ref in zip(filterd_preds, filterd_refs):

        for fp in filterd_pred:
            if fp in filterd_ref:
                true_positive += 1
            else:
                false_positive += 1

        for fr in filterd_ref:
            if not fr in filterd_pred:
                false_negative += 1
                
    # https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L282
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive) 
    else:
        precision = 0

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return precision, recall, f1, bleu_score



def calculate_results(refs, preds, target_vocab=None):
    #calc precision, recall and F1
    #same as https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L239
    
    filterd_refs = [ref[:ref.index(EOS)] for ref in refs]
    filterd_preds = [pred[:pred.index(EOS)] if EOS in pred else pred for pred in preds]
    if target_vocab is not None:
        refs_in_words = [[target_vocab.id2word[r] for r in ref if r not in special_tokens] for ref in refs]
        preds_in_words = [[target_vocab.id2word[p] for p in pred if p not in special_tokens] for pred in preds]
        print(f'BLEU = {get_nltk_bleu_score_for_corpora(refs_in_words, preds_in_words)}')
        print(f'Target = {refs_in_words[:10]}')
        print(f'Predicted = {preds_in_words[:10]}')

    true_positive, false_positive, false_negative = 0, 0, 0

    for filterd_pred, filterd_ref in zip(filterd_preds, filterd_refs):

        if filterd_pred == filterd_ref:
            true_positive += len(filterd_pred)
            continue

        for fp in filterd_pred:
            if fp in filterd_ref:
                true_positive += 1
            else:
                false_positive += 1

        for fr in filterd_ref:
            if not fr in filterd_pred:
                false_negative += 1
                
    # https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L282
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive) 
    else:
        precision = 0

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return precision, recall, f1


class EarlyStopping(object):
    def __init__(self, filename = None, patience=3, warm_up=0, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.warm_up = warm_up
        self.filename = filename

    def __call__(self, score, model, epoch):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            
        elif (score <= self.best_score) and (epoch > self.warm_up) :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (epoch <= self.warm_up):
                print('Warming up until epoch', self.warm_up)
            
            else:
                if self.verbose:
                    print(f'Score improved. ({self.best_score:.6f} --> {score:.6f}).')
                
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

    def save_checkpoint(self, score, model):
        
        if self.filename is not None:
            torch.save(model.state_dict(), self.filename)
            
        if self.verbose:
            print('Model saved...')
        

def pad_seq(seq, max_length):
    # pad tail of sequence to extend sequence length up to max_length
    res = seq + [PAD for i in range(max_length - len(seq))]
    return res 


def pad_seq_new_tokenizer(seq, max_length):
    # pad tail of sequence to extend sequence length up to max_length
    res = seq + [pad_token for i in range(max_length - len(seq))]
    return res


def calc_bleu(refs, hyps):
    _refs = [[ref[:ref.index(EOS)]] for ref in refs]
    _hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(_refs, _hyps)

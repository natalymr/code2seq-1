import logging
import os
import pickle
import random
from datetime import datetime

import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from commit2seq.code2seq.src.utils import Vocab, EarlyStopping, calculate_results_set, calculate_results
from commit2seq.code2seq.src.common_vars import PAD, BOS, EOS, UNK, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, device
from commit2seq.code2seq.src.model import EncoderDecoder_with_Attention
from commit2seq.code2seq.src.DataLoader import DataLoader


def masked_cross_entropy(logits, target):
    return mce(logits.view(-1, logits.size(-1)), target.view(-1))


def compute_loss(batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S,
                 max_length_N, max_length_E, max_length_Y, lengths_k, index_N, model, optimizer=None,
                 is_train=True):
    model.train(is_train)

    use_teacher_forcing = is_train and (random.random() < teacher_forcing_rate)

    target_max_length = batch_Y.size(0)
    pred_Y = model(batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S,
                   max_length_N, max_length_E, max_length_Y, lengths_k, index_N, target_max_length, batch_Y,
                   use_teacher_forcing)

    loss = masked_cross_entropy(pred_Y.contiguous(), batch_Y.contiguous())

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_Y = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()

    return loss.item(), batch_Y, pred

if __name__ == '__main__':
    config_file = '../configs/config_code2seq.yml'

    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    # Data source
    DATA_HOME = config['data']['home']
    DICT_FILE = DATA_HOME + config['data']['dict']
    TRAIN_DIR = DATA_HOME + config['data']['train']
    VALID_DIR = DATA_HOME + config['data']['valid']
    TEST_DIR = DATA_HOME + config['data']['test']

    # Training parameter
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    lr = config['training']['lr']
    teacher_forcing_rate = config['training']['teacher_forcing_rate']
    nesterov = config['training']['nesterov']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    decay_ratio = config['training']['decay_ratio']
    save_name = config['training']['save_name']
    warm_up = config['training']['warm_up']
    patience = config['training']['patience']

    # Model parameter
    token_size = config['model']['token_size']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    bidirectional = config['model']['bidirectional']
    rnn_dropout = config['model']['rnn_dropout']
    embeddings_dropout = config['model']['embeddings_dropout']
    num_k = config['model']['num_k']

    # etc
    slack_url_path = config['etc']['slack_url_path']
    info_prefix = config['etc']['info_prefix']

    # ==================================================================================================================
    torch.manual_seed(1)
    random_state = 42

    run_id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    log_file = '../logs/' + run_id + '.log'
    exp_dir = '../runs/' + run_id

    os.mkdir(exp_dir)

    logging.basicConfig(format='%(asctime)s | %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file,
                        level=logging.DEBUG)
    # ==================================================================================================================
    # load vocab dict
    with open(DICT_FILE, 'rb') as file:
        subtoken_to_count = pickle.load(file)
        node_to_count = pickle.load(file)
        target_to_count = pickle.load(file)
        max_contexts = pickle.load(file)
        num_training_examples = pickle.load(file)
    # ==================================================================================================================
    # making vocab dicts for terminal subtoken, nonterminal node and target.
    word2id = {
        PAD_TOKEN: PAD,
        BOS_TOKEN: BOS,
        EOS_TOKEN: EOS,
        UNK_TOKEN: UNK,
    }

    vocab_subtoken = Vocab(word2id=word2id)
    vocab_nodes = Vocab(word2id=word2id)
    vocab_target = Vocab(word2id=word2id)

    vocab_subtoken.build_vocab(list(subtoken_to_count.keys()), min_count=0)
    vocab_nodes.build_vocab(list(node_to_count.keys()), min_count=0)
    vocab_target.build_vocab(list(target_to_count.keys()), min_count=0)

    vocab_size_subtoken = len(vocab_subtoken.id2word)
    vocab_size_nodes = len(vocab_nodes.id2word)
    vocab_size_target = len(vocab_target.id2word)

    num_length_train = num_training_examples
    # ==================================================================================================================
    mce = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD)
    # ==================================================================================================================
    batch_time = False
    train_dataloader = DataLoader(TRAIN_DIR, batch_size, num_k,
                                  vocab_subtoken, vocab_nodes, vocab_target, device=device,
                                  batch_time=batch_time, shuffle=True)
    valid_dataloader = DataLoader(VALID_DIR, batch_size, num_k,
                                  vocab_subtoken, vocab_nodes, vocab_target, device=device,
                                  shuffle=False)

    model_args = {
        'input_size_subtoken': vocab_size_subtoken,
        'input_size_node': vocab_size_nodes,
        'output_size': vocab_size_target,
        'hidden_size': hidden_size,
        'token_size': token_size,
        'bidirectional': bidirectional,
        'num_layers': num_layers,
        'rnn_dropout': rnn_dropout,
        'embeddings_dropout': embeddings_dropout,
        'device': device
    }

    model = EncoderDecoder_with_Attention(**model_args).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov = nesterov)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)

    fname = exp_dir + save_name
    early_stopping = EarlyStopping(fname, patience, warm_up, verbose=True)
    # ==================================================================================================================


    # ==================================================================================================================
    #
    # Training Loop
    #
    progress_bar = False  # progress bar is visible in progress_bar = False

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.
        train_refs = []
        train_hyps = []
        valid_loss = 0.
        valid_refs = []
        valid_hyps = []

        # train
        for batch in tqdm(train_dataloader,
                          total=train_dataloader.num_examples // train_dataloader.batch_size + 1,
                          desc='TRAIN'):
            batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, \
            max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = batch

            loss, gold, pred = compute_loss(
                batch_S, batch_N, batch_E, batch_Y,
                lengths_S, lengths_N, lengths_E, lengths_Y,
                max_length_S, max_length_N, max_length_E, max_length_Y,
                lengths_k, index_N, model, optimizer,
                is_train=True
            )

            train_loss += loss
            train_refs += gold
            train_hyps += pred

        # valid
        for batch in tqdm(valid_dataloader,
                          total=valid_dataloader.num_examples // valid_dataloader.batch_size + 1,
                          desc='VALID'):
            batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, \
            max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = batch

            loss, gold, pred = compute_loss(
                batch_S, batch_N, batch_E, batch_Y,
                lengths_S, lengths_N, lengths_E, lengths_Y,
                max_length_S, max_length_N, max_length_E, max_length_Y,
                lengths_k, index_N, model, optimizer,
                is_train=False
            )

            valid_loss += loss
            valid_refs += gold
            valid_hyps += pred

        train_loss = np.sum(train_loss) / train_dataloader.num_examples
        valid_loss = np.sum(valid_loss) / valid_dataloader.num_examples

        # F1 etc
        train_precision, train_recall, train_f1 = calculate_results_set(train_refs, train_hyps, vocab_target)
        valid_precision, valid_recall, valid_f1 = calculate_results_set(valid_refs, valid_hyps, vocab_target)

        print()
        print(f'epoch = {epoch}, F1 = {valid_f1}')
        print()
        early_stopping(valid_f1, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #     msgr.print_msg('Epoch {}: train_loss: {:5.2f}  train_f1: {:2.4f}  valid_loss: {:5.2f}  valid_f1: {:2.4f}'.format(
        #             epoch, train_loss, train_f1, valid_loss, valid_f1))

        print('-' * 80)

        scheduler.step()
    # ==================================================================================================================
    model = EncoderDecoder_with_Attention(**model_args).to(device)

    fname = exp_dir + save_name
    ckpt = torch.load(fname)
    model.load_state_dict(ckpt)

    model.eval()
    # ==================================================================================================================
    test_dataloader = DataLoader(TEST_DIR, batch_size, num_k, vocab_subtoken, vocab_nodes, vocab_target, device=device,
                                 batch_time=batch_time, shuffle=True)
    refs_list = []
    hyp_list = []

    for batch in tqdm(test_dataloader,
                      total=test_dataloader.num_examples // test_dataloader.batch_size + 1,
                      desc='TEST'):
        batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = batch
        target_max_length = batch_Y.size(0)
        use_teacher_forcing = False

        pred_Y = model(batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S,
                       max_length_N, max_length_E, max_length_Y, lengths_k, index_N, target_max_length, batch_Y,
                       use_teacher_forcing)

        refs = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()[0]
        pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()[0]

        refs_list.append(refs)
        hyp_list.append(pred)
    print('Tested model : ' + fname)

    test_precision, test_recall, test_f1 = calculate_results(refs_list, hyp_list, vocab_target)
    print('Test : precision {:1.5f}, recall {:1.5f}, f1 {:1.5f}'.format(test_precision, test_recall, test_f1))

    test_precision, test_recall, test_f1 = calculate_results_set(refs_list, hyp_list, vocab_target)
    print('Test(set) : precision {:1.5f}, recall {:1.5f}, f1 {:1.5f}'.format(test_precision, test_recall, test_f1))

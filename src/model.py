import torch
import numpy as np
from torch import nn, einsum
from commit2seq.code2seq.src.common_vars import PAD, BOS

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size_subtoken, input_size_node, token_size, hidden_size, bidirectional=True, num_layers=2,
                 rnn_dropout=0.5, embeddings_dropout=0.25):
        """
        input_size_subtoken : # of unique subtoken
        input_size_node : # of unique node symbol
        token_size : embedded token size
        hidden_size : size of initial state of decoder
        rnn_dropout = 0.5 : rnn drop out ratio
        embeddings_dropout = 0.25 : dropout ratio for context vector
        """

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.token_size = token_size

        self.embedding_subtoken = nn.Embedding(input_size_subtoken, token_size, padding_idx=PAD)
        self.embedding_node = nn.Embedding(input_size_node, token_size, padding_idx=PAD)

        self.lstm = nn.LSTM(token_size, token_size, num_layers=num_layers, bidirectional=bidirectional,
                            dropout=rnn_dropout)
        self.out = nn.Linear(token_size * 4, hidden_size)

        self.dropout = nn.Dropout(embeddings_dropout)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

    def forward(self, batch_S, batch_N, batch_E, lengths_k, index_N, lengths_N, hidden=None):
        """
        :param batch_S: (B * k, l) start terminals' subtoken of each ast path
                        (BS * #_of_paths, max_count_of_subtokens)
        :param batch_N: (l, B * k) nonterminals' nodes of each ast path
                        (max_count_of_nodes_in_paths, BS * #_of_paths)
        :param batch_E: (B * k, l) end terminals' subtoken of each ast path
                        (BS * #_of_paths, max_count_of_subtokens)
        :param lengths_k: length of k in each example
        :param index_N: index for unsorting,
        :param lengths_N:
        :param hidden:
        :return:
        """
        bk_size = batch_N.shape[1]

        # (B * k, l, d)
        encode_S = self.embedding_subtoken(batch_S)
        encode_E = self.embedding_subtoken(batch_E)

        # encode_S (B * k, d) token_representation of each ast path
        encode_S = encode_S.sum(1)
        encode_E = encode_E.sum(1)

        """
        LSTM Outputs: output, (h_n, c_n)
        output (seq_len, batch, num_directions * hidden_size)
        h_n    (num_layers * num_directions, batch, hidden_size) : tensor containing the hidden state for t = seq_len.
        c_n    (num_layers * num_directions, batch, hidden_size)
        """

        # emb_N :(l, B*k, d)
        emb_N = self.embedding_node(batch_N)
        packed = pack_padded_sequence(emb_N, lengths_N)
        output, (hidden, cell) = self.lstm(packed, hidden)
        # output, _ = pad_packed_sequence(output)

        # hidden (num_layers * num_directions, batch, hidden_size)
        # only last layer, (num_directions, batch, hidden_size)
        hidden = hidden[-self.num_directions:, :, :]

        # -> (Bk, num_directions, hidden_size)
        hidden = hidden.transpose(0, 1)

        # -> (Bk, 1, hidden_size * num_directions)
        hidden = hidden.contiguous().view(bk_size, 1, -1)

        # encode_N (Bk, hidden_size * num_directions)
        encode_N = hidden.squeeze(1)

        # encode_SNE  : (B*k, hidden_size * num_directions + 2)
        encode_SNE = torch.cat([encode_N, encode_S, encode_E], dim=1)

        # encode_SNE  : (B*k, d)
        encode_SNE = self.out(encode_SNE)

        # unsort as example
        # index = torch.tensor(index_N, dtype=torch.long, device=device)
        # encode_SNE = torch.index_select(encode_SNE, dim=0, index=index)
        index = np.argsort(index_N)
        encode_SNE = encode_SNE[[index]]

        # as is in  https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L511
        encode_SNE = self.dropout(encode_SNE)

        # output_bag  : [ B, (k, d) ]
        output_bag = torch.split(encode_SNE, lengths_k, dim=0)

        # hidden_0  : (1, B, d)
        # for decoder initial state
        hidden_0 = [ob.mean(0).unsqueeze(dim=0) for ob in output_bag]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)

        return output_bag, hidden_0


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, rnn_dropout):
        """
        hidden_size : decoder unit size,
        output_size : decoder output size,
        rnn_dropout : dropout ratio for rnn
        """

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=rnn_dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, seqs, hidden, attn):
        emb = self.embedding(seqs)
        _, hidden = self.gru(emb, hidden)

        output = torch.cat((hidden, attn), 2)
        output = self.out(output)

        return output, hidden


class EncoderDecoder_with_Attention(nn.Module):
    """Conbine Encoder and Decoder"""

    def __init__(self, input_size_subtoken, input_size_node, token_size, output_size, hidden_size, device,
                 bidirectional=True, num_layers=2, rnn_dropout=0.5, embeddings_dropout=0.25):

        super(EncoderDecoder_with_Attention, self).__init__()
        self.device= device
        self.encoder = Encoder(input_size_subtoken, input_size_node, token_size, hidden_size,
                               bidirectional=bidirectional, num_layers=num_layers, rnn_dropout=rnn_dropout,
                               embeddings_dropout=embeddings_dropout)
        self.decoder = Decoder(hidden_size, output_size, rnn_dropout)

        self.W_a = torch.rand((hidden_size, hidden_size), dtype=torch.float, device=device, requires_grad=True)

        nn.init.xavier_uniform_(self.W_a)

    def forward(self, batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N,
                max_length_E, max_length_Y, lengths_k, index_N, terget_max_length, batch_Y=None,
                use_teacher_forcing=False):

        # Encoder
        encoder_output_bag, encoder_hidden = \
            self.encoder(batch_S, batch_N, batch_E, lengths_k, index_N, lengths_N)

        _batch_size = len(encoder_output_bag)
        decoder_hidden = encoder_hidden

        # make initial input for decoder
        decoder_input = torch.tensor([BOS] * _batch_size, dtype=torch.long, device=self.device)
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)

        # output holder
        decoder_outputs = torch.zeros(terget_max_length, _batch_size, self.decoder.output_size, device=self.device)

        # print('=' * 20)
        for t in range(terget_max_length):

            # ct
            ct = self.attention(encoder_output_bag, decoder_hidden, lengths_k)

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, ct)

            # print(decoder_output.max(-1)[1])

            decoder_outputs[t] = decoder_output

            # Teacher Forcing
            if use_teacher_forcing and batch_Y is not None:
                decoder_input = batch_Y[t].unsqueeze(0)
            else:
                decoder_input = decoder_output.max(-1)[1]

        return decoder_outputs

    def attention(self, encoder_output_bag, hidden, lengths_k):

        """
        encoder_output_bag : (batch, k, hidden_size) bag of embedded ast path
        hidden : (1 , batch, hidden_size):
        lengths_k : (batch, 1) length of k in each example
        """

        # e_out : (batch * k, hidden_size)
        e_out = torch.cat(encoder_output_bag, dim=0)

        # e_out : (batch * k(i), hidden_size(j))
        # self.W_a  : [hidden_size(j), hidden_size(k)]
        # ha -> : [batch * k(i), hidden_size(k)]
        ha = einsum('ij,jk->ik', e_out, self.W_a)  # просто матричное умножение e_out и W_a

        # ha -> : [batch, (k, hidden_size)]
        ha = torch.split(ha, lengths_k, dim=0)

        # dh = [batch, (1, hidden_size)]
        hd = hidden.transpose(0, 1)
        hd = torch.unbind(hd, dim=0)

        # _ha : (k(i), hidden_size(j))
        # _hd : (1(k), hidden_size(j))
        # at : [batch, ( k(i) ) ]
        at = [F.softmax(torch.einsum('ij,kj->i', _ha, _hd), dim=0) for _ha, _hd in zip(ha, hd)]

        # a : ( k(i) )
        # e : ( k(i), hidden_size(j))
        # ct : [batch, (hidden_size(j)) ] -> [batch, (1, hidden_size) ]
        ct = [torch.einsum('i,ij->j', a, e).unsqueeze(0) for a, e in zip(at, encoder_output_bag)]

        # ct [batch, hidden_size(k)]
        # -> (1, batch, hidden_size)
        ct = torch.cat(ct, dim=0).unsqueeze(0)

        return ct
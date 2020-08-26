import torch
from torch import nn, einsum
from commit2seq.code2seq.src.common_vars import BOS
from commit2seq.code2seq.src.model import Encoder, Decoder

import torch.nn.functional as F


class Commit2Seq(nn.Module):
    """Conbine Encoder and Decoder"""

    def __init__(self, input_size_subtoken, input_size_node, token_size, output_size, hidden_size, device,
                 bidirectional=True, num_layers=2, rnn_dropout=0.5, embeddings_dropout=0.25):

        super(Commit2Seq, self).__init__()
        self.device = device
        self.encoder = Encoder(input_size_subtoken, input_size_node, token_size, hidden_size,
                               bidirectional=bidirectional, num_layers=num_layers, rnn_dropout=rnn_dropout,
                               embeddings_dropout=embeddings_dropout)
        self.decoder = Decoder(hidden_size, output_size, rnn_dropout)

        self.W_a_del = torch.rand((hidden_size, hidden_size), dtype=torch.float, device=device, requires_grad=True)
        self.W_a_add = torch.rand((hidden_size, hidden_size), dtype=torch.float, device=device, requires_grad=True)
        self.W_a_3 = torch.rand((hidden_size, hidden_size), dtype=torch.float, device=device, requires_grad=True)

        nn.init.xavier_uniform_(self.W_a_del)
        nn.init.xavier_uniform_(self.W_a_add)
        nn.init.xavier_uniform_(self.W_a_3)

    def forward(self, left_leaves_del, nodes_del, right_leaves_del, nodes_del_len, lengths_k_del, perm_index_del,
                left_leaves_add, nodes_add, right_leaves_add, nodes_add_len, lengths_k_add, perm_index_add,
                target_max_length,
                target=None, use_teacher_forcing=False):

        # Encoder del
        encoder_output_bag_del, encoder_hidden_del = \
            self.encoder(left_leaves_del, nodes_del, right_leaves_del, lengths_k_del, perm_index_del, nodes_del_len)

        # Encoder del
        encoder_output_bag_add, encoder_hidden_add = \
            self.encoder(left_leaves_add, nodes_add, right_leaves_add, lengths_k_add, perm_index_add, nodes_add_len)

        _batch_size = len(encoder_output_bag_add)
        decoder_hidden = (encoder_hidden_del + encoder_hidden_add) / 2

        # make initial input for decoder
        decoder_input = torch.tensor([BOS] * _batch_size, dtype=torch.long, device=self.device)
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)

        # output holder
        decoder_outputs = torch.zeros(target_max_length, _batch_size, self.decoder.output_size, device=self.device)

        def concat_del_add(context_del, context_add):
            def squeeze_0_dim(x):
                return x.squeeze(dim=0)

            ct_del_splitted = tuple(map(squeeze_0_dim, torch.split(context_del, 1, dim=1)))
            ct_add_splitted = tuple(map(squeeze_0_dim, torch.split(context_add, 1, dim=1)))

            result = ()
            for del_, add_ in zip(ct_del_splitted, ct_add_splitted):
                result = result + (torch.cat((del_, add_), 0), )
            return result

        # print('=' * 20)
        for t in range(target_max_length):
            # ct
            ct_del = self.attention(encoder_output_bag_del, decoder_hidden, lengths_k_del, self.W_a_del)
            ct_add = self.attention(encoder_output_bag_add, decoder_hidden, lengths_k_add, self.W_a_add)

            ct = self.attention(concat_del_add(ct_del, ct_add), decoder_hidden, [2] * _batch_size, self.W_a_3)
            # ct = ct_del + ct_add

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, ct)

            decoder_outputs[t] = decoder_output

            # Teacher Forcing
            if use_teacher_forcing and target is not None:
                decoder_input = target[t].unsqueeze(0)
            else:
                decoder_input = decoder_output.max(-1)[1]

        return decoder_outputs

    def attention(self, encoder_output_bag, hidden, lengths_k, attentions_weights):

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
        ha = einsum('ij,jk->ik', e_out, attentions_weights)

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

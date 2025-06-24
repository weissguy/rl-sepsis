# implementation of lstm autoencoder from https://github.com/matanle51/LSTM_AutoEncoder/blob/master/models/LSTMAE.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, seq_len):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.dropout = dropout
        self.seq_len = seq_len

        self.lstm_enc = nn.LSTM(input_dim, hidden_dim, dropout=dropout, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (last_h_state, last_c_state) = self.lstm_enc(x_packed)
        out = pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        
        x_enc = self.fc_enc(last_h_state.squeeze(dim=0))
        
        return x_enc, out


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, seq_len):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.dropout = dropout
        self.seq_len = seq_len

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.lstm_dec = nn.LSTM(hidden_dim, hidden_dim, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        # we don't need to worry about padding here b/c it's excluded from the encoder output
        decoder_input = self.fc_dec(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, (hidden_state, cell_state) = self.lstm_dec(decoder_input)
        dec_out = self.out(dec_out)

        return dec_out, hidden_state


class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, dropout_ratio=0.0):
        super(LSTMAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_ratio, seq_len)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, dropout_ratio, seq_len)

    def forward(self, x, lengths, return_last_h=False, return_enc_out=False):
        x_enc, enc_out = self.encoder(x, lengths)
        x_dec, last_h = self.decoder(x_enc)

        if return_last_h:
            return x_dec, last_h
        elif return_enc_out:
            return x_dec, enc_out
        return x_dec

# -*- coding: utf-8 -*-
import warnings 
warnings.simplefilter('ignore')
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=1, hid_dim_1=32, n_layers=1, dropout=.1):
        super().__init__()
        
        self.hid_dim_1 = hid_dim_1
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hid_dim_1, n_layers, dropout = dropout, batch_first=True)
        
    def forward(self, src):
        
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim=1, hid_dim_1=32, hid_dim_2=64, n_layers=1, dropout=.1):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim_1 = hid_dim_1
        self.n_layers = n_layers
        self.hid_dim_2 = hid_dim_2
        
        self.rnn = nn.LSTM(hid_dim_1, hid_dim_2, n_layers, dropout = dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim_2, output_dim)
        
    def forward(self, input, hidden, cell):
        
        input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, input_dim=1, hid_dim_1=32, 
                 n_layers=1, dropout=.1, device="cpu",
                 output_dim=1, hid_dim_2=64):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hid_dim_1, n_layers, dropout)
        self.decoder = Decoder(output_dim, hid_dim_1, hid_dim_2, n_layers, dropout)
        self.device = device
        
        assert self.encoder.hid_dim_1 == self.decoder.hid_dim_1, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        out_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, out_dim).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        outputs[:, trg_len -1, :] = self.decoder.
        #first input to the decoder is the <sos> tokens
        input = trg[0, :]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = np.random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
# -*- coding: utf-8 -*-
import warnings 
warnings.simplefilter('ignore')

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname('../*'))
sys.path.append(os.path.dirname('src/'))
from deep_svdd.src.optim.ae_trainer import AETrainer
from deep_svdd.src.deepSVDD import DeepSVDD


class AutoEncoder(nn.Module) :
    def __init__(self, rep_dim=10, input_shape=30, dim = 128):
        super().__init__()
        self.relu = nn.ReLU()
        
        ## Encoder
        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(input_shape, dim,  bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(dim, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)
        ## Decoder
        self.fc3 = nn.Linear(self.rep_dim, dim,  bias=False)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.bn2 = nn.BatchNorm1d(dim, eps=1e-04, affine=False)
        self.fc4 = nn.Linear(dim,  input_shape,  bias=False)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        encode = self.fc2(self.relu(self.fc1(x)))
        decode = self.fc4(self.relu(self.fc3(encode)))
        return encode, decode
    
class FeatureExtractor(nn.Module) :
    def __init__(self, rep_dim=10, input_shape=30, dim = 128):
        super().__init__()
        self.relu = nn.ReLU()
        self.dim = dim
        self.input_shape = input_shape
        ## Encoder
        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(input_shape, dim,  bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(dim, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        encode = self.fc2(self.relu(self.fc1(x)))
        return None, encode 

class AE_SCORER :
    def __init__(self, rep_dim=10, input_shape=30, dim = 128, mode ="AutoEncoder", 
                 n_epochs=40, batch_size=128, path_model="./model/", device = "cpu",
                 weight_decay=1e-6) :
        self.ae = AutoEncoder(rep_dim, input_shape, dim)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.path_model = path_model
        self.mode = mode
        self.device = device
        self.weight_decay = weight_decay
        
    def fit (self, X_train, verbose=False) :
        ae_trainer =  AETrainer(device=self.device, n_epochs=self.n_epochs,
                                weight_decay=self.weight_decay)
        
        train_loader = torch.utils.data.DataLoader(torch.FloatTensor(X_train), 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True, num_workers=0)
        self.ae = ae_trainer.train(train_loader, self.ae,
                                   verbose=verbose)
            
        torch.save(self.ae.state_dict(), self.path_model + self.mode + str(self.ae.rep_dim))
    def decision_function(self, X_val) :
        
        encode, decode = self.ae(torch.FloatTensor(X_val))
        scores = np.linalg.norm(decode.data.detach().cpu().numpy() - X_val, axis=1)
        return scores
    def load_model(self) :
        self.ae.load_state_dict(torch.load(self.path_model + self.mode + str(self.ae.rep_dim)))
        return self.ae
        
class DSVDD :
    def __init__(self, rep_dim=10, input_shape=30, dim = 128, mode ="pretrain_ae", 
                 n_epochs=40, batch_size=128, weight_decay=0.1,
                 path_model="./model/", device ="cpu", runs=None, output=None) :
        assert mode in ("pretrain_ae", "train_only", "AutoEncoder")
        self.mode = mode
        self.net = FeatureExtractor(rep_dim, input_shape, dim)   
                 
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.path_model = path_model
        self.mode = mode
        self.weight_decay = weight_decay
        self.device = device
    def fit(self, X_train, verbose=False) :
        train_loader = torch.utils.data.DataLoader(torch.FloatTensor(X_train), 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True, num_workers=0)
        if self.mode in ["train_only", "pretrain_ae"] :
            
            if self.mode == "train_only" :
                self.D_SVDD = DeepSVDD(net = self.net,
                                     ae_net = None)
            else :
                mode_ = "AutoEncoder"
                
                ae_scorer = AE_SCORER(self.net.rep_dim, self.net.input_shape, self.net.dim,
                                      mode_, self.n_epochs, self.batch_size, self.path_model,
                                      self.device, self.weight_decay)
                try :
                    #import pdb; pdb.set_trace()
                    self.ae = ae_scorer.load_model()
                except :
                    ae_scorer.fit(X_train, verbose=verbose)
                    self.ae = ae_scorer.ae
                    
                self.D_SVDD = DeepSVDD(net = self.net,
                                     ae_net = self.ae,)             
            
                self.D_SVDD.init_network_weights_from_pretraining() 
        
            self.D_SVDD.train(train_loader, device=self.device, 
                          weight_decay= self.weight_decay,
                        n_epochs=self.n_epochs, verbose=verbose)
        
        else :
            mode_ = "AutoEncoder"
            ae_scorer = AE_SCORER(self.net.rep_dim, self.net.input_shape, self.net.dim,
                                  mode_, self.n_epochs, self.batch_size, self.path_model,
                                  self.device, self.weight_decay)
            ae_scorer.fit(X_train, verbose=verbose)
            self.ae = ae_scorer.ae
            
        
    def decision_function(self, X_val) :
        if self.mode in ["train_only", "pretrain_ae"]  :
            _, encode = self.D_SVDD.net(torch.FloatTensor(X_val).to(self.device))
            scores = np.linalg.norm(encode.data.cpu().detach().numpy() - self.D_SVDD.c, axis=1)
        else :
            encode, decode = self.ae(torch.FloatTensor(X_val).to(self.device))
            scores = np.linalg.norm(decode.data.detach().cpu().numpy() - X_val, axis=1)    
        return scores
        
        
        
        
        
        
        
        
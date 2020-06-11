#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:25:12 2020

@author: jores
"""

import warnings 
warnings.simplefilter('ignore')
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname('../*'))
sys.path.append(os.path.dirname('src/'))
from deep_svdd.src.optim.ae_trainer import AETrainer
from deep_svdd.src.deepSVDD import DeepSVDD
from utilities import  plot_precision_recall, \
  writeResults, Dataset, visualize_doc_embeddings
from Iso_Oneclass.iso_oneclass import train_and_predict
    
class AutoEncoder(nn.Module) :
    def __init__(self, rep_dim):
        super().__init__()
        self.relu = nn.ReLU()
        
        ## Encoder
        self.rep_dim = rep_dim
        dim = 128
        self.fc1 = nn.Sequential(nn.Linear(30, dim,  bias=False),
                                 self.relu)
        self.fc2 = nn.Sequential(nn.Linear(dim, self.rep_dim, bias=False), 
                                 self.relu)
        ## Decoder
        self.fc3 = nn.Sequential(nn.Linear(self.rep_dim, dim,  bias=False),
                                 self.relu)
        self.fc4 = nn.Linear(dim, 30,  bias=False)
        
    def forward(self, x):
        encode = self.fc2(self.fc1(x))
        decode = self.fc4(self.fc3(encode))
        return encode, decode
    
class FeatureExtractor(nn.Module) :
    def __init__(self, rep_dim):
        super().__init__()
        self.relu = nn.ReLU()
        dim = 128 
        ## Encoder
        self.rep_dim = rep_dim
        self.fc1 = nn.Sequential(nn.Linear(30, dim,  bias=False),
                                 self.relu)
        self.fc2 = nn.Sequential(nn.Linear(dim, self.rep_dim,  bias=False), 
                                 self.relu)
        
    def forward(self, x):
        encode = self.fc2(self.fc1(x))
        return None, encode 

data = Dataset()

data.X_train = data.X_train[np.where(data.Y_train == 1)[0]] 
data.Y_train = data.Y_train[np.where(data.Y_train == 1)[0]] 
weight_decay = {2:0.1, 5: 1, 10:1}

for rep_dim in [2, 5, 10, 20] :
    auc_dict = {#"AE" : {"rec" : [], "iso" : [], "oneclass": []},
                "DSVDD" : {"rec" : [], "iso" : [], "oneclass": []}
                }
    print(" ")
    print("Dimension :", rep_dim)    
    for n_run in range(5) :        
        
        ### Training the Autoencoder and visualizing 
        nb_epochs = 40
        print("Auto Encoder")
        auto_encoder = AutoEncoder(rep_dim)
        try :
            auto_encoder.load_state_dict(torch.load("model/" + str(n_run) + "_final_AutoEncoder_" + str(rep_dim)  ))
        except :
            ae_trainer =  AETrainer(device='cpu', n_epochs=40)
            auto_encoder = ae_trainer.train(data, auto_encoder)
            
            torch.save(auto_encoder.state_dict(), "model/" + str(n_run) + "_final_AutoEncoder_" + str(rep_dim))
        
#        encode, decode = auto_encoder(torch.FloatTensor(data.X_val))
#        encode_train, _ = auto_encoder(torch.FloatTensor(data.X_train))
#        scores = np.linalg.norm(decode.data.detach().cpu().numpy() - data.X_val, axis=1)
#        
#        visualize_doc_embeddings(encode.data.detach().cpu().numpy(),
#                              ['blue','red',], 
#                              data.Y_val, str(n_run) + "_final_AE_"+ str(rep_dim), center=None)
#        
#        auc_rec = plot_precision_recall([scores], [data.Y_val], name = ['Val'],
#                                  name_model = str(n_run) + "_final_AutoEncoder_" + str(rep_dim))
#
#        auc_iso = train_and_predict("Iforest", encode_train.data.detach().cpu().numpy(),
#                          encode.data.detach().cpu().numpy(), data.Y_val, 
#                          name=str(n_run) + "_final_AE_Iforest_bis" + str(rep_dim), save=False)
#        
#        auc = train_and_predict("OneClass", encode_train.data.detach().cpu().numpy()[:50000], 
#                          encode.data.detach().cpu().numpy(), data.Y_val, 
#                          name=str(n_run) + "_final_AE_OneClass_bis_" + str(rep_dim),
#                          contamination=0.001, save=False)
#        
#        auc_dict["AE"]["rec"].append(auc_rec)
#        auc_dict["AE"]["iso"].append(auc_iso)
#        auc_dict["AE"]["oneclass"].append(auc)
        
        print("Deep SVDD....")
                
#        ### Training the DeepSVDD 
#        
        # auto_encoder = AutoEncoder(rep_dim)
        network = FeatureExtractor(rep_dim)
        
        ## With pretraining
        deep_svdd = DeepSVDD(net = network, #objective='soft-boundary',
                             ae_net = auto_encoder)
        #deep_svdd.pretrain(data, device = 'cpu', n_epochs=30)
        deep_svdd.init_network_weights_from_pretraining()
        deep_svdd.train(data, device='cpu', weight_decay = weight_decay[rep_dim],
                        n_epochs=nb_epochs)
        ### Analyzing the results
        
        _, encode = deep_svdd.net(torch.FloatTensor(data.X_val))
        _, encode_train = deep_svdd.net(torch.FloatTensor(data.X_train))
        scores = np.linalg.norm(encode.data.detach().numpy() - deep_svdd.c, axis=1)
        
        visualize_doc_embeddings(encode.data.detach().cpu().numpy(),
                              ['blue','red',], 
                              data.Y_val, str(n_run) + "_pretrain_DSVDD_"+ str(rep_dim), center=deep_svdd.c)
        
        auc_rec = plot_precision_recall([scores], [data.Y_val], name = ['Val'],
                                  name_model = str(n_run) + "_pretrain_DSVDD_" + str(rep_dim))
        
        auc_iso = train_and_predict("Iforest", encode_train.data.detach().cpu().numpy(),
                          encode.data.detach().cpu().numpy(), data.Y_val, 
                          name=str(n_run) + "_pretrain_SVDD_Iforest_bis_" + str(rep_dim), save=False)
        auc = train_and_predict("OneClass", encode_train.data.detach().cpu().numpy()[:50000], 
                          encode.data.detach().cpu().numpy(), data.Y_val, 
                          name=str(n_run) + "_pretrain_SVDD_OneClass_" + str(rep_dim),
                          contamination=0.001, save=False)
        auc_dict["DSVDD"]["rec"].append(auc_rec)
        auc_dict["DSVDD"]["iso"].append(auc_iso)
        auc_dict["DSVDD"]["oneclass"].append(auc) 
        
        
    for elt, value in auc_dict.items():
        for elt_, value_ in value.items()    :
            name = "_0_pretrain" + elt + "_" + elt_ + "_"
            auc = np.mean(value_)
            std_auc = np.std(value_)
            writeResults(name + str(rep_dim), dim=rep_dim,
                     auc=auc, path = "./results/auc_performance.csv",
                     std_auc = std_auc, nb_epochs=nb_epochs)
    import json
    with open( "./results/auc_" + str(rep_dim)+".json", "w") as file :
        json.dump(auc_dict, file)






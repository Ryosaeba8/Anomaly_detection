# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
from utils import aucPerformance
import os
sys.path.append(os.path.dirname('../*'))
import argparse
from utilities import Dataset
from network import DevNet

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="unsupervised", help="the mode of the algorithm")
parser.add_argument("--hidden_dim", type=int, default=20, help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=256, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=50, help="the number of batches per epoch")
parser.add_argument("--n_epochs", type=int, default=30, help="the number of epochs")
parser.add_argument("--runs", type=int, default=10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--output", type=str, default='./results/performance_global.csv', help="the output file path")
parser.add_argument("--known_outliers", type=int, default=10, help="the number of labeled outliers available at hand")


args = parser.parse_args()

data = Dataset(mode="other")

def get_train(data, args) :

    x_train, y_train = data.X_train, data.Y_train

    if mode =="unsupervised" :
        x_train = x_train[np.where(y_train == 0)[0]]   
    
    if mode == "semi_supervised":
        outlier_ids = np.where(y_train == 1)[0]
        outlier_to_keep = np.random.choice(outlier_ids, int(args.known_outliers/2))
        outlier_to_remove = np.setdiff1d(outlier_ids, outlier_to_keep)
        x_train = np.delete(x_train, outlier_to_remove, axis=0)
        y_train = np.delete(y_train, outlier_to_remove, axis=0)
    return x_train, y_train
    
def train_and_save(args, data) :
    rauc = np.zeros(args.runs)
    ap = np.zeros(args.runs) 
    if args.mode == "unsupervised" :
        real_ano = 0
    if args.mode == "semi_supervised" :
        real_ano = int(args.known_outliers/2)
    if args.mode =="supervised" :
        real_ano = args.known_outliers
    print("Mode :", args.mode, "Total outliers :",
           known_outliers, "Real outliers", real_ano)
    for i in range(args.runs) :
        x_train, y_train = get_train(data, args)
        devnet = DevNet(**args.__dict__)
        devnet.fit(x_train, y_train)
        scores = devnet.decision_function(data.X_val)
        rauc[i], ap[i] = aucPerformance(scores, data.Y_val) 
        

    mean_aucpr = np.mean(ap)
    std_aucpr = np.std(ap)
    df = pd.DataFrame.from_dict(args.__dict__, orient ="index").T
    df["real_anomaly"] = real_ano
    df["mean_auc"] = mean_aucpr
    df["std_auc"] = std_aucpr
    try :
        pd.read_csv(args.output)
        df.to_csv(args.output, mode="a", index=False, header=False)
    except :
       df.to_csv(args.output, mode="a", index=False)  
       

for mode in ["semi_supervised", "supervised"] :
    args.mode = mode   
        
    for known_outliers in range(10, 400, 20) :
        args.known_outliers = known_outliers
        train_and_save(args, data) 

    
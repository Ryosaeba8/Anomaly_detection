import numpy as np
import pandas as pd
import sys
from utils import aucPerformance
import os
sys.path.append(os.path.dirname('../*'))
import argparse
from utilities import Dataset
from dsvdd_network  import  DSVDD

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="AutoEncoder", help="the mode of the algorithm")
parser.add_argument("--rep_dim", type=int, default=20, help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=256, help="batch size used in SGD")
parser.add_argument("--n_epochs", type=int, default=40, help="the number of epochs")
parser.add_argument("--runs", type=int, default=5, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--output", type=str, default='./results/performance_no_bn.csv', help="the output file path")
parser.add_argument("--weight_decay", type=float, default=0.05, help="the regularization weight")
parser.add_argument("--device", type=str, default="cuda", help="the regularization weight")


args = parser.parse_args()

data = Dataset(mode="other")
data.X_train = data.X_train[np.where(data.Y_train == 0)]
    
def train_and_save(args, data) :
    rauc = np.zeros(args.runs)
    ap = np.zeros(args.runs) 

    print("Mode :", args.mode, "Rep dim :", args.rep_dim, "L2 weight :",
          args.weight_decay)
    for i in range(args.runs) :
        d_svdd = DSVDD(**args.__dict__)
        d_svdd.fit(data.X_train, verbose=False)
        scores = d_svdd.decision_function(data.X_val)
        rauc[i], ap[i] = aucPerformance(scores, data.Y_val) 
        

    mean_aucpr = np.mean(ap)
    std_aucpr = np.std(ap)
    df = pd.DataFrame.from_dict(args.__dict__, orient ="index").T
    df["mean_auc"] = mean_aucpr
    df["std_auc"] = std_aucpr
    try :
        pd.read_csv(args.output)
        df.to_csv(args.output, mode="a", index=False, header=False)
    except :
       df.to_csv(args.output, mode="a", index=False)  
       


for rep_dim in [5, 10, 20, 30] :
    args.rep_dim = rep_dim
    for weight in  [0.00001, 0.01, 0.1]:
        args.weight_decay = weight
        for mode in ["AutoEncoder", "pretrain_ae", "train_only"] :
            args.mode = mode  
            train_and_save(args, data) 
            
            
            
            
            
            
            
            
            
            
# -*- coding: utf-8 -*-


import numpy as np
import warnings; warnings.simplefilter('ignore')
import os
import sys
sys.path.append(os.path.dirname('../*'))
from utilities import Dataset, plot_precision_recall, writeResults
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.spatial import cKDTree
plt.ioff()
    
def train_and_predict(model, X_train, X_val, Y_val,
                      name, contamination = "", save=True) :
    
    if model =="KNN":
         tree = cKDTree(X_train)
         scores, ind = tree.query(X_val, k=contamination, n_jobs=8) 
         list_neigh = np.arange(100, contamination, 50)
         for neig in list_neigh :
             scores_ = np.mean(scores[:, :neig], axis=1)
             auc = plot_precision_recall([scores_], [Y_val], name = ['Val'],
                          name_model = "hard_KNN_"+ str(neig))
             writeResults(name, dim=neig,
                     auc=auc, path = "./results/auc_performance.csv",
                     std_auc = 0.0)      
         return None 
    
    elif model == "Iforest" : 
        model = IsolationForest(n_estimators=100)  
        model.fit(X_train)
        scores = -model.score_samples(X_val)
    elif model =="OneClass" :
        model = OneClassSVM(gamma='auto',
                            nu=contamination)  
        model.fit(X_train)
        scores = -model.score_samples(X_val)
    
    auc = plot_precision_recall([scores], [Y_val], name = ['Val'],
                              name_model = name)
    if save :
        writeResults(name, dim=contamination,
                     auc=auc, path = "./results/auc_performance.csv",
                     std_auc = 0.0)
    return auc;

if __name__ == '__main__' : 
    
    data = Dataset()
    X_train = data.X_train[np.where(data.Y_train == 1)[0]]   
    
    
    ## Training KNN 
    contamination = 1000
    train_and_predict("KNN", data.X_train, data.X_val, data.Y_val, 
                      "KNN", contamination = contamination, save=False)
    
    ## Training Iforest
    auc = np.zeros(10)
    for n_run in range(auc.shape[0]) :
        auc[n_run] = train_and_predict("Iforest", X_train, 
                                data.X_val, data.Y_val, 
                                "Iforest", contamination = "", save=False)
    writeResults("IsolationForest", dim="",
                 auc=np.mean(auc), path = "./results/auc_performance.csv",
                 std_auc = np.std(auc))   
    
    ## Training IsolationForest
    for contamination in [0.0001, 0.001, 0.01, 0.1,] :
        train_and_predict("OneClass", X_train[:50000], data.X_val, data.Y_val,
                          "OneClass_"+str(contamination), contamination = contamination)
































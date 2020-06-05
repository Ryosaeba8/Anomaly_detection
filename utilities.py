#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang

Source code for the REPEN algorithm in KDD'18. See the following paper for detail.
Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu. 2018. Learning Representations
of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. 
In KDD 2018: 24th ACM SIGKDD International Conferenceon Knowledge Discovery & 
Data Mining, August 19–23, 2018, London, UnitedKingdom.

"""

import pandas as pd
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
cmap = plt.get_cmap("tab10") 
plt.style.use('ggplot')
import warnings; warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split

def visualize_doc_embeddings(my_doc_embs, my_colors, my_labels, my_name, center):
    my_pca = PCA(n_components=2)
    #my_tsne = TSNE(n_components=2, perplexity=10) #https://lvdmaaten.github.io/tsne/
    doc_embs_pca = my_pca.fit_transform(my_doc_embs) 
    
    doc_embs_tsne = doc_embs_pca #my_tsne.fit_transform(doc_embs_pca)
    
    fig, ax = plt.subplots()
    
    for i, label in enumerate(list(set(my_labels))):
        idxs = [idx for idx, elt in enumerate(my_labels) if elt==label]
        ax.scatter(doc_embs_tsne[idxs,0], 
                   doc_embs_tsne[idxs,1], 
                   cmap = cmap(i),
                   label=str(label),
                   alpha=0.7,
                   s=40)
    if center is not None :
        center = my_pca.transform(np.array(center)[np.newaxis, :])
        ax.scatter(center[:, 0], center[:, 1], c ='black')
    ax.legend(scatterpoints=1)
    fig.suptitle(my_name,
                 fontsize=10)
    fig.set_size_inches(11,7)
    fig.savefig('images/pca_' + my_name + '.png')

def cutoff(values, th = 1.7321):
    sorted_indices = np.argsort(values, axis=0)
#    print(sorted_indices)
    values = values[sorted_indices, 0]
#    print(values)
    v_mean = np.mean(values)
    v_std = np.std(values)
    th = v_mean + th * v_std #1.7321 
#    print(th)
    outlier_ind = np.where(values > th)[0]
    inlier_ind = np.where(values <= th)[0]
#    print(sorted_indices[np.where(sorted_indices == outlier_ind)])
    outlier_ind = sorted_indices[outlier_ind]
    inlier_ind = sorted_indices[inlier_ind]
#    print(outlier_ind)
    #print(labels[ind])
    return inlier_ind, outlier_ind;
#    return outlier_ind, inlier_ind;

def cutoff_unsorted(values, th = 1.7321):
#    print(values)
    v_mean = np.mean(values)
    v_std = np.std(values)
    th = v_mean + th * v_std #1.7321 
    if th >= np.max(values): # return the top-10 outlier scores
        temp = np.sort(values)
        th = temp[-11]
    outlier_ind = np.where(values > th)[0]
    inlier_ind = np.where(values <= th)[0]
    return inlier_ind, outlier_ind;

def writeResults(name, dim, auc, path = "./results/auc_performance.csv", std_auc = 0.0):    
    csv_file = open(path, 'a') 
    row = name + "," + str(dim)+ "," + str(auc) + "," + str(std_auc) + "\n"
    csv_file.write(row)


class Dataset() :
    def __init__(self, path_data = '../data/creditcard.csv') :
        df = pd.read_csv(path_data)
        for col in df.columns[:-1] :
            df[col] = (df[col] - df[col].mean())/df[col].std()
        
        df.loc[df.Class == 1, 'Class'] = -1
        df.loc[df.Class == 0, 'Class'] = 1
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(df.drop(columns = ['Class']).values, 
                                                         df.Class.values, test_size=0.2,
                                                         random_state=42, 
                                                        stratify = df.Class.values)    
    def create_data_loader(self, batch_size):
        import torch
        self.train_loader = torch.utils.data.DataLoader(torch.FloatTensor(self.X_train), 
                                                        batch_size=batch_size, 
                                                        shuffle=True, num_workers=0)
        
        self.val_loader = torch.utils.data.DataLoader(torch.FloatTensor(self.X_val),
                                                      batch_size=batch_size, 
                                                      shuffle=True, num_workers=0)
        
def plot_precision_recall(list_outlier_scores, label, name,
                          name_model, mode='min_max') :
    plt.figure()
    for i, outlier_scores in enumerate (list_outlier_scores) :
        if mode == 'min_max' :
            scores_tr = (outlier_scores - outlier_scores.min())/(outlier_scores.max() - outlier_scores.min())
        else :
           scores_tr = outlier_scores 
        pr, recall, thr = precision_recall_curve(label[i], scores_tr, pos_label=-1)
        auc_pr_rec = auc(recall, pr)
        plt.plot(recall[:-1], pr[:-1], label = " AUC " + name[i] + " : " + str(round(auc_pr_rec, 4)))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(name_model)
    plt.legend(loc='best')
    plt.savefig('images/' + name_model + '.png')
    print("AUC PR-REC :", auc_pr_rec)    
    return auc_pr_rec
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname('../*'))
from DEVNET.utils import aucPerformance
from sklearn.ensemble import IsolationForest
from DEVNET.network import DevNet
from sklearn.model_selection import train_test_split
from keras.models import Model
from utilities import visualize_doc_embeddings
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
plt.style.use("ggplot")

## Génération d'une mixture de k gaussiennes
def generate_mix_gauss(k=2,  p=2, seed=123, nb=10000, plot=False, prop=0.008, coef=12):
    n_samples = nb
    np.random.seed(seed)
    labels = []
    
    # generate zero centered stretched Gaussian data
    C = np.eye(p)
    ind = np.random.randint(p)
    C[ind] = 1.*np.ones(p)
    #C = np.array([[1., 0], [1, 1]])
    gauss_1 = np.dot(np.random.randn(n_samples, p), C)
    labels += [1]*len(gauss_1)
    
    if k == 1 :
        mini, maxi = gauss_1.min(axis=0), gauss_1.max(axis=0)
        for i in range(p) :
            center_i = np.zeros(p)
            center_i[i] = mini[i]
            #import pdb; pdb.set_trace()
            center_i[:i] = maxi[:i]
            center_i[i+1:] = maxi[i+1:]
            out_i = np.random.normal(loc=center_i-3, scale=1.5, size=(int((1/p)*n_samples*prop), p))
            gauss_1 = np.vstack([gauss_1, out_i])
            labels += [-1]*len(out_i)
        X = gauss_1
    else : 
        gauss_2 = np.random.randn(n_samples, p) + coef*np.ones(p)
        labels += [0]*len(gauss_2)
        maxi_1 = gauss_1.max(axis=0); mini_2 = gauss_2.min(axis=0)
        out_center = np.mean(np.vstack([maxi_1, mini_2]), axis=0)
        out = np.random.normal(loc=out_center, scale=1.5, size=(int(n_samples*prop), p))
        gauss_2 = np.vstack([gauss_2, out])
        labels += [-1]*len(out)
        X = np.vstack([gauss_1, gauss_2])
    out_1 = np.random.normal(loc=np.array([-2, 10]), scale=1., size=(10, p))
    out_2 = np.random.normal(loc=np.array([8, -2]), scale=1., size=(10, p))
    out = np.vstack([out_1, out_2])
    X = np.vstack([X, out])
    labels += [-1]*len(out)
    # concatenate the two datasets into the final training set
    
    if (plot) & (p==2) :
        fig, ax = plt.subplots(1,1, figsize =(8, 8), sharex = True, sharey = True)
        ax.scatter(X[:, 0].T, X[:, 1].T,
                      c=labels,
                       **plot_kwds)
    shuffle = np.arange(len(X))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    labels = np.array(labels)[shuffle] 
    return X, labels

X, labels = generate_mix_gauss(k=2,  p=2, seed=123, coef=8,
                               nb=50000, plot=True, prop=0.01)

labels[np.where(labels !=-1)] = 0
labels[np.where(labels ==-1)] = 1

X_train, X_val, y_train, y_val = train_test_split(X, 
                                                  labels, test_size=0.2,
                                                  random_state=42, 
                                                  stratify =labels)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_val = (X_val - mean)/std


visualize_doc_embeddings(X_val, ['blue','red'], y_val, "raw synthetic data_2", None)  


### Isolation Forest ###
model = IsolationForest(n_estimators=100)  
model.fit(X_train)
scores = -model.score_samples(X_val)
auc = aucPerformance(scores, y_val) 

## Devnet
devnet = DevNet(known_outliers=50, hidden_dim = 10, n_neighbors=4,
                n_epochs=30, mode="unsupervised", nb_batch=100)

lesinn_scores = devnet.lesinn(X_train, X_val)
auc = aucPerformance(lesinn_scores, y_val)
devnet.fit(X_train, y_train, verbose=True)
devnet_scores = devnet.decision_function(X_val)
auc = aucPerformance(devnet_scores, y_val) 

intermediate_layer = Model(devnet.network.model.input, 
                           devnet.network.model.get_layer('hl1').output)

repres = intermediate_layer.predict(X_val)

y_pred = np.where(devnet_scores >2, 1, 0)
visualize_doc_embeddings(repres, ['blue','red'], y_pred[:, 0], "devnet_intermediate_synth_2_", None)

visualize_doc_embeddings(X_val, ['blue','red'], y_pred[:, 0], "devnet_intermediate_synth_2_", None)

x_normal = X_val[np.where(y_val == 0)]
x_anormal = X_val[np.where(y_val == 1)]
w1, b1, w2, b2 = devnet.network.model.get_weights()


tmp = pd.DataFrame(np.maximum(0, x_normal[:4].dot(w1) + b1))
tmp["label"] = "normal"

tmp2 = pd.DataFrame(np.maximum(0, x_anormal[:4].dot(w1) + b1))
tmp2["label"] = ["anormal_1", "anormal_1", "anormal_2", "anormal_2"]

tmp = tmp.append(tmp2)
tmp_one_mixture = tmp.copy()









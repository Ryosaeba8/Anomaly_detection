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
from sklearn.ensemble import RandomForestClassifier


#df_ = pd.read_csv("data/kddcup.data_10_percent_corrected", header = None)
#with open("data/kddcup.names", "r") as f :
#    lines = f.readlines()
#    
#df_.columns =[line.split(":")[0] for line in lines[1:] + ["label"]]

df = pd.read_csv("data/kdd99-unsupervised-ad.tab", header=None, sep="\t")
#df = df.drop(columns=[8])
df[29] = pd.get_dummies(df[29], drop_first=True)
#for col in df.columns[:-1] :
#    df[col] = (df[col] - df[col].mean())/df[col].std()

X_train, X_val, y_train, y_val = train_test_split(df.drop(columns = [29]).values, 
                                                  df[29].values, test_size=0.2,
                                                  random_state=42, 
                                                  stratify = df[29].values)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_val = (X_val - mean)/std

visualize_doc_embeddings(X_val, ['blue','red'], y_val, "raw data", None)   

### Isolation Forest ###
model = IsolationForest(n_estimators=100)  
model.fit(X_train)
scores = -model.score_samples(X_val)
auc = aucPerformance(scores, y_val) 


#### DEVNET ####

devnet = DevNet(input_shape=29, known_outliers=1,
                n_epochs=50, mode="supervised")
devnet.fit(X_train, y_train, verbose=True)
scores = devnet.decision_function(X_val)
auc = aucPerformance(scores, y_val) 

intermediate_layer = Model(devnet.network.model.input, 
                           devnet.network.model.get_layer('hl1').output)

repres = intermediate_layer.predict(X_val)
visualize_doc_embeddings(repres, ['blue','red'], y_val, "devnet_intermediate_10", None)


clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_val)[:, 1]
auc = aucPerformance(y_pred, y_val) 


#X_sim = np.random.multivariate_normal(np.rand(30), np.)




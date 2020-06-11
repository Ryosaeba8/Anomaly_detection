# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('ggplot')

## KNN and Isolation Forest
df = pd.read_csv("Iso_Oneclass/results/auc_performance.csv")
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2) # 2x2 grid
ax0 = fig.add_subplot(gs[0, 0]) # first row, first col
ax1 = fig.add_subplot(gs[0, 1]) # first row, second col
ax2 = fig.add_subplot(gs[1, :]) # full second row

df[df.algo =="KNN"].plot(x = "param", y="auc_mean", ax=ax0, title="KNN_WITHOUT_ANOMALIES")
df[df.algo =="KNN_HARD"].plot(x = "param", y="auc_mean", ax=ax1, title="KNN_WITH_ANOMALIES")
ax2.table(cellText=df.values[:5], colLabels=df.columns, loc='center', alpha = .8, 
          fontsize=250)
ax2.axis('off')
ax2.axis('tight')
fig.savefig("KNN_ISO_OneClass.png")

### REPEN
df = pd.read_csv("REPEN/results/auc_performance.csv")
fig, ax = plt.subplots(2, 1, figsize=(14, 10))
ax[0].errorbar(df[df.Nom == "supervised"].Dimension, df[df.Nom == "supervised"].auc_mean,
            yerr=df[df.Nom == "supervised"].auc_std, fmt='-o', label="Supervised")
ax[0].errorbar(df[df.Nom == "Unsupervised"].Dimension, df[df.Nom == "Unsupervised"].auc_mean,
            yerr=df[df.Nom == "Unsupervised"].auc_std, fmt='-o', label="Unsupervised")
ax[1].errorbar(df[df.Nom == "semi_supervised"].Dimension, df[df.Nom == "semi_supervised"].auc_mean,
            yerr=df[df.Nom == "semi_supervised"].auc_std, fmt='-o', label="Semi supervised")
ax[0].legend(loc="best")
ax[0].set_xlabel("DIMENSION OF THE LATENT REPRESENTATION")

ax[1].legend(loc="best")
ax[1].set_xlabel("NB ANOMALIES")
ax[0].set_ylabel("AUC PRECISION RECALL")
plt.title("REPEN_DIMENSION_10")
fig.savefig("REPEN.png")

### DEVIATION NETWORK
df = pd.read_csv("DEVNET/results/performance.csv").sort_values(by=["nb_utilise_anomalie"])
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.errorbar(df[(df.dataset == "fraud_2") & (df.contamination == 0.05) ].nb_utilise_anomalie, 
            df[(df.dataset == "fraud_2") & (df.contamination == 0.05) ].mean_auc_pr,
            yerr=df[(df.dataset == "fraud_2") & (df.contamination == 0.05) ].std_auc_pr, 
            fmt='-o', label="SEMI SUPERVISED WITH CONTAMINATION")

ax.errorbar(df[(df.dataset == "fraud_2") & (df.contamination == 0) ].nb_utilise_anomalie, 
            df[(df.dataset == "fraud_2") & (df.contamination == 0) ].mean_auc_pr,
            yerr=df[(df.dataset == "fraud_2") & (df.contamination == 0) ].std_auc_pr, 
            fmt='-o', label="SEMI SUPERVISED WITHOUT CONTAMINATION")

ax.errorbar(df[(df.dataset == "fraud_lesinn")  ].nb_utilise_anomalie, 
            df[(df.dataset == "fraud_lesinn") ].mean_auc_pr,
            yerr=df[(df.dataset == "fraud_lesinn")  ].std_auc_pr, 
            fmt='-o', label="UNSUPERVISED WITHOUT CONTAMINATION")
plt.legend(loc="best")
plt.xlabel("NUMBER OF ANOMALIES USED TO TRAIN THE MODEL")
plt.ylabel("AUC PRECISION RECALL")
plt.title("DEVIATION NETWORK")
fig.savefig("DEVNET.png")


### DEEP_SVDD

## Remarques gros pbs de stabilité et de dégénerescence
## Grande Dimension regulariser à bloc
## Faible DImension regulariser légèrement
df = pd.read_csv("deep_svdd/results/auc_performance.csv").sort_values(by="params")
df = df.dropna()
df.Nom = df.Nom.apply(lambda x: " ".join(x.split("_")[-3:-1]))
fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
ax[0].errorbar(df[(df.Nom == "AE rec") & (df.regularisation == 0) ].params, 
            df[(df.Nom == "AE rec") & (df.regularisation == 0)].auc_mean,
            yerr=df[(df.Nom == "AE rec") & (df.regularisation == 0)].auc_std, 
            fmt='-o', label="AutoEncoder")

ax[1].errorbar(df[(df.Nom == "DSVDD rec") & (df.regularisation == 0) ].params, 
            df[(df.Nom == "DSVDD rec") & (df.regularisation == 0)].auc_mean,
            yerr=df[(df.Nom == "DSVDD rec") & (df.regularisation == 0)].auc_std, 
            fmt='-o', label="DSVDD with low regularisation")

ax[2].errorbar(df[(df.Nom == "DSVDD rec") & (df.regularisation == 1) ].params, 
            df[(df.Nom == "DSVDD rec") & (df.regularisation == 1)].auc_mean,
            yerr=df[(df.Nom == "DSVDD rec") & (df.regularisation == 1)].auc_std, 
            fmt='-o', label="DSVDD with high regularisation")
ax[0].legend(loc="best"); ax[1].legend(loc="best"); ax[2].legend(loc="best")
plt.xlabel("Dimension of latent space")
plt.ylabel("AUC PRECISION RECALL")
plt.title("Deep SVDD")
fig.savefig("Deep_SVDD.png")






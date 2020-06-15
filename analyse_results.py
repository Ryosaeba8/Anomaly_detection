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


df = pd.read_csv("REPEN/results/performance_global.csv").sort_values(by=["known_outliers"])

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
for mode in df["mode"].unique() :
    label = " ".join(mode.split("_")).upper()
    if mode =="semi_supervised" :
        label += " (HALF ARE REAL ANOMALIES)"
    ax.errorbar(df[(df["mode"] == mode) ].known_outliers, 
                df[(df["mode"] == mode) ].mean_auc,
                yerr=df[(df["mode"] == mode)].std_auc, 
                fmt='-o', label=label)
ax.legend(loc="best")
ax.set_xlabel("NB OF CANDIDATES ANOMALIES")
ax.set_ylabel("AUC PRECISION RECALL")
plt.title("REPEN")
fig.savefig("REPEN_FINAL.png")

### DEVIATION NETWORK
df = pd.read_csv("DEVNET/results/performance_global.csv").sort_values(by=["known_outliers"])

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
for mode in df["mode"].unique() :
    label = " ".join(mode.split("_")).upper()
    if mode =="semi_supervised" :
        label += " (HALF ARE REAL ANOMALIES)"
    ax.errorbar(df[(df["mode"] == mode) ].known_outliers, 
                df[(df["mode"] == mode) ].mean_auc,
                yerr=df[(df["mode"] == mode)].std_auc, 
                fmt='-o', label=label)
ax.legend(loc="best")
ax.set_xlabel("NB OF CANDIDATES ANOMALIES")
ax.set_ylabel("AUC PRECISION RECALL")
plt.title("DEVNET")
fig.savefig("DEVNET_FINAL.png")


### DEEP_SVDD

## Remarques gros pbs de stabilité et de dégénerescence
## Grande Dimension regulariser à bloc
## Faible DImension regulariser légèrement
df = pd.read_csv("deep_svdd/results/performance_1.csv")

df = df[df.rep_dim.isin([5])]
fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
for i, mode in enumerate(df["mode"].unique()) :
    df_tmp = df[df["mode"] == mode]
    for weight in df_tmp.rep_dim.unique() :
        df_tmp_bis = df_tmp[df_tmp.rep_dim == weight]
        ax[i].errorbar(df_tmp_bis.weight_decay, 
            df_tmp_bis.mean_auc,
            yerr=df_tmp_bis.std_auc, 
            fmt='-o', label=mode+"_"+str(weight))
    ax[i].legend(loc="best")
    ax[i].set_title(mode)

ax[i].set_ylabel("AUC PRECISION RECALL")
ax[i].set_xlabel("Weight decay")

plt.title("Deep SVDD")
fig.savefig("Deep_SVDD.png")






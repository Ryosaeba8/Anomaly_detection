# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 22:14:41 2021

@author: Jores
"""
import pandas as pd
df_jess = pd.read_excel("logement_jess.xlsx")
df_joel = pd.read_excel("logement.xlsx")

df_jess["in_joel"] = df_jess.lien.apply(lambda x : x.split()[0] in df_joel.url.tolist())
df_inter = df_jess[df_jess.in_joel]
df_jess_not_in_joel = df_jess[~df_jess.in_joel]
df_joel_not_in_jess = df_joel[~df_joel.url.isin(df_inter.lien.str.split().str[0].tolist())]

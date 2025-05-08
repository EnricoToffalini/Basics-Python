
#############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#############################

# import 

inv1 = pd.read_csv("../data/ExerData_Invalsi_1.csv",sep=";")
inv2 = pd.read_csv("../data/ExerData_Invalsi_2.csv")
wc = pd.read_excel("../data/ExerData_Wechsler.xlsx")
lt = pd.read_csv("../data/ExerData_LabTrials.csv")
qu = pd.read_csv("../data/ExerData_Questionnaires.csv")

#############################

# invalsi
inv = pd.concat([inv1,inv2])
cols = ["InvalsiRead" + str(i) for i in range(1, 13)]
inv["InvalsiTot"] = inv[cols].sum(1)
inv.loc[inv["InvalsiTot"]>200,"InvalsiTot"] = np.nan

# wechsler
for c in wc.columns[2:wc.shape[1]]:
  wc[c] = pd.to_numeric(wc[c].astype(str).str.replace(",","."), errors="coerce")
cols = wc.columns[2:wc.shape[1]]
wc["WechslerTot"] = wc[cols].sum(1)
wc.loc[wc["WechslerTot"]>200,"WechslerTot"] = np.nan

# lab trials
lt = lt.groupby("idName").sum(0).reset_index()

# qu
cols = ["Openness" + str(i) for i in range(1, 9)]
for c in cols:
  qu[c] = qu[c].astype(str).str.replace(",",".")
  qu[c] = qu[c].astype(str).str.replace("_",".")
  qu[c] = qu[c].astype(str).str.replace("/",".")
  qu[c] = pd.to_numeric(qu[c], errors="coerce")
qu["OpennessTot"] = qu[cols].sum(1)
cols = ["Agreab" + str(i) for i in range(1, 9)]
for c in cols:
  qu[c] = qu[c].astype(str).str.replace(",",".")
  qu[c] = qu[c].astype(str).str.replace("_",".")
  qu[c] = qu[c].astype(str).str.replace("/",".")
  qu[c] = pd.to_numeric(qu[c], errors="coerce")
qu["AgreabTot"] = qu[cols].sum(1)

#############################

df = inv[["idName","gender","InvalsiTot"]].merge(wc[["idName","WechslerTot"]], on="idName", how="outer")
df = df.merge(lt[["idName","respAcc"]], on="idName", how="outer")
df = df.merge(qu[["idName","OpennessTot","AgreabTot"]], on="idName", how="outer")

#############################

cols = ["InvalsiTot","WechslerTot","respAcc","OpennessTot","AgreabTot"]
df[cols].corr().round(2)
df.describe()
df.describe().round(2).loc[["mean","std"]]
df[cols].skew(0).round(2)
df[cols].kurt(0).round(2)

stats.ttest_ind(df["InvalsiTot"][df["gender"]=="M"].dropna(), df["InvalsiTot"][df["gender"]=="F"].dropna())

plt.hist(df["WechslerTot"],bins=50)
plt.show(); plt.clf()


#############################



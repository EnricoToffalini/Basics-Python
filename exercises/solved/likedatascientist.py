
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("../data/df1.csv").drop(columns=["Unnamed: 0"])
dfWide1 = pd.read_csv("../data/dfWide1.csv").drop(columns=["Unnamed: 0"])
dfWide2 = pd.read_csv("../data/dfWide2.csv").drop(columns=["Unnamed: 0"])

##########################

whichnumeric = []
nrows = df1.shape[0]
for c in df1.columns:
  x = pd.to_numeric(df1[c],errors="coerce")
  howmanyNaNs = x.isna().sum()
  if (howmanyNaNs/nrows) < 0.80:
      df1[c] = x
      whichnumeric.append(c)

df1[whichnumeric].corr(method="pearson").round(3)
df1[whichnumeric].dropna().corr(method="pearson").round(3)
df1[whichnumeric].shape
df1[whichnumeric].dropna().shape

sns.heatmap(df1[whichnumeric].corr(method="pearson").round(3))
plt.show()
plt.clf()

##########################

dfLong1 = pd.melt(
    dfWide1,
    id_vars = ["id","age","group"],
    var_name = "Time",
    value_name = "Score",
    value_vars = ["T0", "T1", "T2"]
)

sns.lineplot(data=dfLong1, x="Time", y="Score",
    hue="id", units="id")
plt.show()
plt.clf()

sns.pointplot(data=dfLong1, x="Time",  y="Score",
    hue="group", errorbar="ci", dodge=True, join=True)
plt.show()
plt.clf()

##########################

dfLong2_acc = pd.melt(
    dfWide2,
    id_vars = ["id","age","group"],
    var_name = "Time",
    value_name = "Acc",
    value_vars = ["Acc_T0", "Acc_T1", "Acc_T2"]
)
dfLong2_acc["Time"] = dfLong2_acc["Time"].str.extract(r"T(\d)").astype(int)
# alternatively, using list comprehension
# dfLong2_acc["Time"] = ["T0" if x == "Acc_T0" else "T1" if x == "Acc_T1" else "T2" for x in dfLong2_acc.Time]

dfLong2_rt = pd.melt(
    dfWide2,
    id_vars = ["id","age","group"],
    var_name = "Time",
    value_name = "RT",
    value_vars = ["RT_T0", "RT_T1", "RT_T2"]
)
dfLong2_rt["Time"] = dfLong2_rt["Time"].str.extract(r"T(\d)").astype(int)
# alternatively, using list comprehension
# dfLong2_rt["Time"] = ["T0" if x == "RT_T0" else "T1" if x == "RT_T1" else "T2" for x in dfLong2_rt.Time]

dfLong2 = pd.merge(dfLong2_acc, dfLong2_rt, on = ["id","age","group","Time"])
dfLong2


##########################

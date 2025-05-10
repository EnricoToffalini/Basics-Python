
import numpy as np
import matplotlib.pyplot as plt

##

for i in range(20):
  print(np.random.logistic(size=1).round(3))

##

for i in range(20):
    x = np.random.normal(size=1)
    if x < 0: 
        print(x)
    else:
        print("positive")

##

x = np.array(np.random.normal(size=10),dtype="float")
np.where(x<0,"neg","pos")

##

niter = 2000
N = 30
res = np.empty(niter,dtype="float")*np.nan
for i in range(niter):
  x = np.random.normal(size=N)
  y = np.random.normal(size=N)
  res[i] = np.corrcoef(x,y)[0,1]

np.median(res)
plt.hist(res,bins=30)
plt.show()

np.quantile(res, q=[0.025, 0.975])

## 

def describe_sign(x):
    if x < 0:
       return "negative"
    elif x == 0:
       return "zero"
    else:
       return "positive"

describe_sign(1.5)
describe_sign(0)
describe_sign(-1.5)

## 

def simulate_correlations(N, n_sim):
    res = np.empty(n_sim)*np.nan
    for i in range(n_sim):
          x = np.random.normal(size=N)
          y = np.random.normal(size=N)
          res[i] = np.corrcoef(x,y)[0,1]
    return res

## 

def simulate_correlations(N, n_sim, r):
  res = np.empty(n_sim)*np.nan
  for i in range(n_sim):
    X = np.random.multivariate_normal(mean=[0,0],cov=[[1,r],[r,1]],size=N)
    res[i] = np.corrcoef(X[:,0],X[:,1])[0,1]
  return res

##

scores = np.random.normal(size=20)
conditions = [scores < -1, (scores > -1) & (scores < 1), scores > 1]
labels = ["low", "average", "high"]
np.select(conditions, labels, default="")

# alternative with list comprehension
['low' if x < -1 else 'high' if x > 1 else 'average' for x in scores] 



import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Independent data
nsamp = 100
u = np.random.uniform(-1,5,nsamp)

# Data:
beta = np.array([4.342, -0.7, 2.7])
y0 = beta[0] + beta[1] * np.exp(-u/2) + beta[2] * np.exp(-u)
wstd = 0.4 # noise
y = y0 + np.random.normal(0,wstd,nsamp)

# Split
utr, uts, ytr, yts = train_test_split(u, y, test_size=0.5)

# Fit:
MSE = []
dtest = np.arange(1,12)

for d in dtest:
    # model using train data:
    Xtr = np.exp(utr[:, None] * np.arange(1, d+1)/(-d))
    model = linear_model.LinearRegression()
    model.fit(Xtr, ytr)

    # predict with test data:
    Xts = np.exp(uts[:, None] * np.arange(1, d+1)/(-d))
    yhat = model.predict(Xts)

    # find mse:
    MSE.append(np.mean((yhat - yts)**2))

print("select d = ", MSE.index(min(MSE)) + 1)

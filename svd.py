# !/usr/bin/env python
"""This script prompts a mathematical approch on Principle Component Analysis"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Reading the database
df = pd.read_csv('..\\Database_for_dimensionality_reduction.csv')
dfcopy = df.select_dtypes(include=[np.number])
print("This is the imported database.")
dfcopy.head()

# Data Imputation
for column in dfcopy:
    meani = dfcopy[column].mean()
    dfcopy[column].fillna(meani, inplace=True)

df_act = dfcopy

# Data Standardization
sc = StandardScaler()
df_scale = sc.fit_transform(df_act)
df_scale = pd.DataFrame(df_scale, columns=df_act.columns)
print("This is the scaled database.")
df_scale.head()

# Data Splitting in the Label and Feature Database

# This is Feature/target variable
Y_label = df_scale["AP1"]
df_scale.drop("AP1", axis=1, inplace=True)
# This is label database.
X_label = df_scale
print("This is label database")
X_label.head()

# SVD on Feature Database

temp1 = X_label.T.dot(X_label)
temp2 = X_label.dot(X_label.T)

eigval_V, eigvec_V = np.linalg.eig(temp1)
eigval_U, eigvec_U = np.linalg.eig(temp2)

temp3 = eigval_V ** 0.5

Cum = temp3 / temp3.sum() * 100
Cum = Cum.cumsum()
# Scree plot of cumulative sum
fig = plt.figure()

x = range(1, len(Cum)+1)
plt.plot(x, Cum)
plt.gca().invert_yaxis()
plt.xlabel("k")
plt.ylabel("Cum. Sum of Eigenvalues")
plt.show()

# taking 60% of the actual data
k = 0
for i in range(len(Cum) - 1):
    if Cum[i] >= 60:
        k = i + 1
        break
print(k)

Sigma = np.zeros((X_label.shape[0], X_label.shape[1]))
Sigma[:X_label.shape[1], :X_label.shape[1]] = np.diag(temp3)

# The value of k is 5
Vt_m = eigvec_V.T[:5, :]
Sigma = Sigma[:, :5]

X_label_Recon_m = eigvec_U.dot(Sigma.dot(Vt_m))
X_label_Recon_m = pd.DataFrame(X_label_Recon_m.real, columns=X_label.columns)

X_label_Trans_m = X_label_Recon_m.dot(Vt_m.T)
X_label_Trans_m.head()

# validation using TruncatedSVD function in sklearn

svd = TruncatedSVD(n_components=5)
svd.fit(X_label)
result = svd.transform(X_label)
result = pd.DataFrame(result)
result.columns = ["PC1", "PC2", "PC3", "PC4", "PC5"]
result["X_label"] = Y_label
print("This is data for validation.")
result.head()

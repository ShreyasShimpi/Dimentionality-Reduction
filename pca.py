# !/usr/bin/env python
"""This script prompts a mathematical approch on Principle Component Analysis"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# PCA on Feature Database

X_cov = X_label.T.dot(X_label) / (len(df) - 1)
eig_val, eig_vec = np.linalg.eig(X_cov)

Sum = eig_val / sum(eig_val) * 100
cum_sum = np.cumsum(Sum)
# Scree plot of cumulative sum
fig = plt.figure()

x = range(1, len(cum_sum)+1)
plt.plot(x, cum_sum)
plt.gca().invert_yaxis()
plt.xlabel("k")
plt.ylabel("Cum. Sum of Eigenvalues")
plt.show()

# taking 85% of the actual data
k = 0
for i in range(len(cum_sum)):
    if cum_sum[i] >= 85:
        k = i + 1
        break
print(k)

# top 5 eigenvalues since k is equal to 5
X_5 = eig_vec[0:5, :]
X_tran = X_label.dot(X_5.T)
X_tran.columns = ["PC1", "PC2", "PC3", "PC4", "PC5"]
X_tran["label"] = Y_label
print("This is the transformed database into 5 Principle Components.")
X_tran.head()

# validation using PCA function in sklearn

pca = PCA(n_components=5)
pca.fit(X_label)
X_pca = pca.transform(X_label)
X_pca = pd.DataFrame(X_pca)
X_pca["X_label"] = Y_label
X_pca.columns = ["PC1", "PC2", "PC3", "PC4", "PC5", "X_label"]
print("This is data for validation.")
X_pca.head()

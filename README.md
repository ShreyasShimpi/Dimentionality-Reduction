# Dimentionality Reduction using PCA and SVD

## Description
Dimensionality Reduction is the process of reducing the dimensions of the feature dataset which have fewer but significant information about the actual dataset
These are the two ways to it, first is we can do feature selection in which we selectively choose the features manually
Second way is to do the feature extraction on the dataset which gives more informative data with less dimentions 
We are going to look at feature extraction techniques such as pca and svd short for Principle component and Singular Value Decomposition 
In this session, We will look at the complete mathematical approach of dimentionality reduction techniques PCA and SVD.

## Prerequisites

The modules we need to install for doing PCA and SVD are pandas , numpy and matplotlib and sklearn. 
Pandas is useful when handelling the database. 
Numpy will be used for matrix operations
Matplotlib will be used for displaying graphs and visuals.
Scikit-Learn library is used for machine learning algorithms.
They can be installed using ```pip```.
```
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
```
## Databse Information
Columns of the dataset: 
	1. Latitude, Longitude, Elevation(m), Distance from Coast (km)
	2. AP1,AP2 - Atmospheric Pressure
	3. DT1,DT2 - Daily Temperature (Day & Night)
	4. HTM,LTM - Highest & Lowest Temperature
	5. AC1,AC2,LC1,LC2 - All clouds and Low Clouds
	6. RFL - Rainfall
	7. RH1,RH2 - Relative Humidity
	8. WSD - Wind Speed

There are 417 rows and 18 columns and index is Names of the states.
The Data have some blank spaces which can be imputed with the respective mean values.
Then we split the data into target variable and feature variable.


## Principle Component Analysis
### Import Packages
first we will import the necessary packages required.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
### Database Introduction
The database is imported and now for dimentionality reduction we will only use the number columns.
```python
df = pd.read_csv('Database_for_dimensionality_reduction.csv')
dfcopy = df.select_dtypes(include=[np.number])
print("This is the imported database.")
dfcopy.head()
```
### Data Imputation
The database imputation is done to replace the missing values with the mean or median of the dataset. In this case, we have imputed the database with the mean of the columns.
```python
for column in dfcopy:
    meani = dfcopy[column].mean()
    dfcopy[column].fillna(meani, inplace=True)

df_act = dfcopy
```
### Data Standerdization
Data Standardization is used to convert all the feature variables to similar scale or common format and centralizes the data.
It should be done to remove anomalies due to large and small scaled variables
The common method to do Data Standardization is Z score

```python
sc = StandardScaler()
df_scale = sc.fit_transform(df_act)
df_scale = pd.DataFrame(df_scale, columns=df_act.columns)
```

### Splitting the Database
The dataset is splitted in feature (Y_label) which is the target variable "AP1" and label (X_label) which is  remianing dataset.

```python
Y_label = df_scale["AP1"]
df_scale.drop("AP1", axis=1, inplace=True)
X_label = df_scale
```

### PCA on Database
The dimentionality reduction using PCA is explained using mathematical approach.

First we take covarience matrix, then find the eigenvalues and eigenvectors of 

```python
X_cov = (X_label.T).dot(X_label)/ (len(df) -1)
eigval, eigvec = np.linalg.eig(X_cov)
```
Then by taking cumulative sum of eigenvalue we will choose the value of k which is no of principle components. The k is decided in this case to retain 85% of the inforamation of the database.
```python
sum = eigval/sum(eigval)*100
cumsum = np.cumsum(sum)
plt.plot(cumsum)
```
In our case k = 5, Then we will select 5 eigenvectors and take a dot product of them with label dataset. Thats how we will get transformed dataset (X_tran) which is down to 5 principle components.
```python        
X_5 = eigvec[0:5,:]
X_tran = X_label.dot(X_5.T)
X_tran.columns = ["PC1","PC2","PC3","PC4","PC5"]
X_tran["label"] = Y_label
```
### Validation
We can validate our results using PCA function in sklearn decomposition library.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X_label)
X_pca = pca.transform(X_label)
X_pca = pd.DataFrame(X_pca)
X_pca.columns = ["PC1", "PC2", "PC3", "PC4", "PC5", "X_label"]
X_pca["X_label"] = Y_label
```
## Singular Value Decomposition
### Import Packages
first we will import the necessary packages required.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
```
### Database Introduction
The database is imported and now for dimentionality reduction we will only use the number columns.
```python
df = pd.read_csv('Database_for_dimensionality_reduction.csv')
dfcopy = df.copy()
dfcopy1= df.select_dtypes(include=[np.number])
dfcopy1.head()
```
### Data Imputation
The database imputation is done to replace the missing values with the mean or median of the dataset. In this case, we have imputed the database with the mean of the columns.
```python
for column in dfcopy1: 
    meani = dfcopy[column].mean()
    dfcopy1[column].fillna(meani, inplace=True)
```
### Data Standerdization
Data Standardization is used to convert all the feature variables to similar scale or common format and centralizes the data.
It should be done to remove anomalies due to large and small scaled variables
The common method to do Data Standardization is Z score

```python
sc= StandardScaler()
df_scale = sc.fit_transform(df_act)
df_scale = pd.DataFrame(df_scale, columns = df_act.columns)
```

### Splitting the Database
The dataset is splitted in feature (Y_label) which is the target variable "AP1" and label (X_label) which is  remianing dataset.

```python
Y_label = df_scale["AP1"]
df_scale.drop("AP1", axis = 1, inplace = True)
X_label = df_scale
```

### SVD on Database
The dimentionality reduction using SVD is explained using mathematical approach.

First we take dot product label datase with the tranposes, which is V and U, then find the eigenvalues and eigenvectors of V and U

```python
temp1 = X_label.T.dot(X_label)
temp2 = X_label.dot(X_label.T)
eigval_V, eigvec_V = np.linalg.eig(temp1)
eigval_U, eigvec_U = np.linalg.eig(temp2)
temp3 = eigval_V ** 0.5
```
Again the value of the k is taken to be 5.
so we will take first 5 eigenvectors of V anf first 5 values of Singular value matrix Sigma
```python
Cum = temp3 / temp3.sum() * 100
Cum = Cum.cumsum()
plt.plot(Cum)
```
Then we will find that eigenvalues of V and U are same and their root is a set of singular values. Then we will build a m*n matrix and put these singular values in it.
```python
Sigma = np.zeros((X_label.shape[0], X_label.shape[1]))
Sigma[:X_label.shape[1], :X_label.shape[1]] = np.diag(S)
```
Again the vale of the k is taken to be 5 for the 60% accuracy. so we will take first 5 eigenvectors of V anf first 5 values of Singular value matrix Sigma
```python
Vt_m = eigvec_V.T[:5,:]
Sigma = Sigma[:,:5]
```
The database is reconstructed using the formula given below which is U * S * V_t where S and V_t are new matrices created above. Then we can transfor the dataset into 5 columns using the formula X_recon * V_t
```python
X_label_Recon_m = (eigvec_U).dot(Sigma.dot(Vt_m))
X_label_Recon_m = pd.DataFrame(X_label_Recon_m.real, columns = X_label.columns)
X_label_Trans_m = X_label_Recon_m.dot(Vt_m.T)
```
### Validation
We can validate our results using TruncatedSVD function in sklearn decomposition library.
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5)
svd.fit(X_label)
result = svd.transform(X_label)
result = pd.DataFrame(result)
result.columns = ["PC1", "PC2", "PC3", "PC4", "PC5"]
result["X_label"] = Y_label
result.head()
```
This was the dimentionality reduction using PCA and SVD.
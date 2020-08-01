
# coding: utf-8

# In[119]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # IMPUTATION

# In[120]:


df = pd.read_csv('Database_for_dimensionality_reduction.csv')
dfcopy = df.copy()
dfcopy1= df.select_dtypes(include=[np.number])
dfcopy1.head()
#df.columns[2]


# In[121]:


dfcopy1.shape


# In[122]:


dfcopy1.isnull().sum()


# In[123]:


for column in dfcopy1: 
        meani = dfcopy[column].mean()
        dfcopy1[column].fillna(meani, inplace=True)


# In[124]:


dfcopy1.isnull().sum()


# In[125]:


dfcopy1.head()


# In[126]:


df_act = dfcopy1[["AP1","AP2","DT1","DT2","HTM","LTM","AC1","AC2","LC1","LC2","RFL","RH1","RH2","WSD"]]


# In[127]:


df_act.head()


# In[128]:


from sklearn.preprocessing import StandardScaler


# In[129]:


sc= StandardScaler()
df_scale = sc.fit_transform(df_act)
df_scale.shape


# In[130]:


df_scale = pd.DataFrame(df_scale, columns = df_act.columns)
df_scale.head()


# ## Choosing AP1 as target function label

# In[131]:


Y_label = df_scale["AP1"]


# In[132]:


df_scale.drop("AP1", axis = 1, inplace = True)


# In[133]:


X_label = df_scale
X_label.head()


# ## PCA ON DATABASE

# In[ ]:


#X_cov2 = np.cov(X_label.T)


# In[135]:


X_cov = (X_label.T).dot(X_label)/ (len(df) -1)
X_cov


# In[136]:


eigval, eigvec = np.linalg.eig(X_cov)


# In[137]:


eigval


# In[185]:


sum = eigval/sum(eigval)*100


# In[140]:


cumsum = np.cumsum(sum)
cumsum


# In[141]:


plt.plot(cumsum)
plt.show()


# In[142]:


#taking 90% of the actual data
for i in range(len(cumsum)):
    if (cumsum[i] >= 90):
        k = i;
        break
k


# In[198]:


#top 5 eigen values
X_5 = eigvec[0:5,:]
X_5.shape


# In[197]:


X_tran = X_label.dot(X_5.T)


# In[203]:


#X_tran.columns = ["PC1","PC2","PC3","PC4","PC5"]
X_tran["label"] = Y_label
X_tran.head()


# In[222]:


plt.figure(figsize=(8,6))
plt.scatter(X_tran.iloc[:,0], X_tran.iloc[:,1] ,c = X_tran.iloc[:,-1],cmap='rainbow')
plt.show()


# ## MEAN of DT1 and DT2 as a lebel 

# In[147]:


#df_act = dfcopy1[["AP1","AP2","DT1","DT2","HTM","LTM","AC1","AC2","LC1","LC2","RFL","RH1","RH2","WSD"]]
#df_act.head()


# In[148]:


# sc= StandardScaler()
# df_scale = sc.fit_transform(df_act)


# In[149]:


#df_scale = pd.DataFrame(df_scale, columns = df_act.columns)


# In[150]:


# Y_label = (df_scale["DT1"] + df_scale["DT2"])/2


# In[151]:


# df_scale.drop(["DT1","DT2"], axis=1, inplace = True)


# In[152]:


# X_label = df_scale
# X_label.head()


# In[153]:


# X_cov = np.cov(X_label.T)
# eigval, eigvec = np.linalg.eig(X_cov)

# #top 5 eigen values
# X_5 = eigvec[0:5,:]
# X_5.shape

# X_tran = X_label.dot(X_5.T)

# X_tran.columns = ["PC1","PC2","PC3","PC4","PC5"]
# X_tran["Y_label"] = Y_label


# In[154]:


# X_tran.head()


# In[155]:


# plt.figure(figsize=(8,6))
# plt.scatter(X_tran.iloc[:,0], X_tran.iloc[:,-1], c = X_tran.iloc[:,-1],cmap='rainbow')
# plt.show()


# ## with pca library

# In[156]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X_label)
X_pca = pca.transform(X_label)


# In[157]:


X_pca = pd.DataFrame(X_pca)

X_pca["Y_label"] = Y_label
X_pca.head()


# In[158]:


X_tran.head()


# In[159]:


plt.figure(figsize=(8,6))
plt.scatter(X_pca.iloc[:,0], X_pca.iloc[:,1], c = X_pca.iloc[:,-1],cmap='rainbow')
plt.show()


# In[ ]:





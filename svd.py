
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('Database_for_dimensionality_reduction.csv')
dfcopy = df.copy()
dfcopy1= df.select_dtypes(include=[np.number])
dfcopy1.head()


# In[3]:


for column in dfcopy1: 
        meani = dfcopy[column].mean()
        dfcopy1[column].fillna(meani, inplace=True)
dfcopy1.head()


# In[4]:


df_act = dfcopy1[["AP1","AP2","DT1","DT2","HTM","LTM","AC1","AC2","LC1","LC2","RFL","RH1","RH2","WSD"]]
df_act.head()


# ## scaling

# In[5]:


sc= StandardScaler()
df_scale = sc.fit_transform(df_act)
df_scale = pd.DataFrame(df_scale, columns = df_act.columns)
df_scale.head()


# ## splitting into label and feature

# In[6]:


Y_label = df_scale["AP1"]
Y_label[:5]


# In[7]:


df_scale.drop("AP1", axis = 1, inplace = True)
X_label = df_scale
#X_label.head()


# In[8]:


X_label.head()


# # SVD

# In[9]:


U, S, Vt = np.linalg.svd(X_label)


# In[10]:


S


# In[11]:


# Cum = S/S.sum()*100
# Cum = Cum.cumsum()
# k=0
# for i in range(len(Cum)):
#     if (Cum[i] >= 90):
#         k = i;
#         break
# k, Cum


# In[12]:


k = 5


# In[44]:


S1 = np.zeros((X_label.shape[0], X_label.shape[1]))
S1[:X_label.shape[1], :X_label.shape[1]] = np.diag(S)


# In[45]:


S1.shape


# In[15]:


Vt.shape,U.shape


# In[46]:


Vt1 = Vt[:5,:]
S11 = S1[:,:5]
S11.shape, Vt1.shape


# In[59]:


X_label_Reconstruct = U.dot(S11.dot(Vt1))


# In[60]:


X_label_Reconstruct = pd.DataFrame(X_label_Reconstruct, columns = X_label.columns)
X_label_Reconstruct.head()


# In[68]:


X_label_Transformed = X_label_Reconstruct.dot(Vt1.T)
X_label_Transformed.head()


# # SVD MATHEMATICAL

# In[19]:


temp1 = (X_label.T).dot(X_label)
temp2 = (X_label).dot(X_label.T)


# In[20]:


temp1.shape, temp2.shape


# In[21]:


eigval_V, eigvec_V = np.linalg.eig(temp1)
eigval_U, eigvec_U = np.linalg.eig(temp2)


# In[22]:


eigvec_U.shape, eigvec_V.shape


# In[29]:


temp3 = eigval_V**0.5


# In[31]:


Sigma = np.zeros((X_label.shape[0], X_label.shape[1]))
Sigma[:X_label.shape[1], :X_label.shape[1]] = np.diag(temp3)
Sigma.shape


# In[82]:


# we can see that S ans Sigma are same....
#we will take k = 5
Vt_m = eigvec_V.T[:5,:]
Sigma = Sigma[:,:5]
Sigma.shape, V_m.shape


# In[95]:


eigvec_U[:1,:20].real


# In[96]:


U[:1,:20]


# In[99]:


X_label_Recon_m = (eigvec_U).dot(Sigma.dot(Vt_m))
X_label_Recon_m = pd.DataFrame(X_label_Recon_m.real, columns = X_label.columns)
X_label_Recon_m.head()


# In[101]:


X_label_Trans_m = X_label_Recon_m.dot(Vt_m.T)
X_label_Trans_m.head()


# ## SVD with SKlearn

# In[65]:


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5)
svd.fit(X_label)
result = svd.transform(X_label)
result = pd.DataFrame(result)


# # The Results Compared

# In[103]:


result.head()   #This is TruncatedSVD sklearn lib result


# In[66]:


X_label_Transformed.head() #This is using numpy SVD 


# In[102]:


X_label_Trans_m.head()    #This is a Complete Mathematical Approch


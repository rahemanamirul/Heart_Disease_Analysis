#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="whitegrid")


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


df=pd.read_csv(r'C:\Users\hp\OneDrive\Documents\Desktop\heart.csv')
df


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


df.columns


# In[11]:


df['target'].nunique()


# In[12]:


df["target"].unique()


# In[13]:


df["target"].value_counts()


# In[14]:


f, ax = plt.subplots(figsize=(8,6))
ax = sns.countplot(x="target", data=df)
plt.show()


# In[15]:


df.groupby("sex")["target"].value_counts()


# In[16]:


f, ax = plt.subplots(figsize=(8,6))
ax = sns.countplot(x="sex", hue="target", data=df)
plt.show()


# In[17]:


ax = sns.catplot(x="target", col="sex", data=df, kind="count", height=5, aspect=1)


# In[18]:


f, ax = plt.subplots(figsize=(8,6))
ax = sns.countplot(y="target", hue="sex", data=df)
plt.show()


# In[19]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df, palette="Set3")
plt.show()


# we can use plt.bar keywords arguments for a different looks:

# In[20]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df, facecolor=(0,0,0,0),linewidth=5, edgecolor=sns.color_palette("dark", 3))
plt.show()


# In[21]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", hue="fbs", data=df)
plt.show()


# In[22]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target",hue="exang", data=df)
plt.show()


# # Biveriate Analysis

# In[23]:


correlation = df.corr()


# In[24]:


correlation["target"].sort_values(ascending=False)


# In[25]:


# explore cp variable

# cp stand for chest pain type
# first iwill check the number if unique values in cp  variabl


# In[26]:


df["cp"].nunique()


# In[27]:


df["cp"].unique()


# In[28]:


df["cp"].value_counts()


# In[29]:


# visualize the frequency distribution of cp variable


# In[30]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", data=df)
plt.show()


# Frequency distribution of target variable wrt cp

# In[31]:


df.groupby("cp")["target"].value_counts()


# In[32]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp",hue="target", data=df)
plt.show()


# In[33]:


ax = sns.catplot(x="target", col="cp", data=df, kind="count", height=8, aspect=1)


# In[34]:


# Analysis of target and thalach variable


# In[35]:


df["thalach"].nunique()


# In[36]:


df["thalach"].unique()


# In[38]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
ax = sns.distplot(x, bins=10)
plt.show()


# In[40]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.distplot(x, bins=10)
plt.show()


# In[44]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
ax = sns.distplot(x, bins=10, vertical=True)
plt.show()


# In[45]:


# seaborn kernel Density Estimation (KDE)Plot


# In[47]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.kdeplot(x)
plt.show()


# In[48]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.kdeplot(x, shade=True, color="r")
plt.show()


# # Histogram

# In[50]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
ax = sns.distplot(x, kde=False, rug=True, bins=10)
plt.show()


# In[52]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df)
plt.show()


# In[53]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df, jitter = 0.01)
plt.show()


# In[54]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="thalach", data=df)
plt.show()


# ### Findings of Bivariate Analysis <a class="anchor" id="8.4"></a>
# 
# Findings of Bivariate Analysis are as follows –
# 
# 
# - There is no variable which has strong positive correlation with `target` variable.
# 
# - There is no variable which has strong negative correlation with `target` variable.
# 
# - There is no correlation between `target` and `fbs`.
# 
# - The `cp` and `thalach` variables are mildly positively correlated with `target` variable. 
# 
# - We can see that the `thalach` variable is slightly negatively skewed.
# 
# - The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
# 
# - The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering from heart disease (target = 0).
# 

# #  multivariante analysis

# In[56]:


plt.figure(figsize=(16,12))
plt.title("correlation Heatmap of Heart Disease Dataset")
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor="white")
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=90)
plt.show()


# # pair plot

# In[57]:


num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]
sns.pairplot(df[num_var], kind="scatter", diag_kind="hist")
plt.show()


# In[58]:


# Analysis of age and other variables


# In[59]:


df["age"].nunique()


# # view statical summary of age variable

# In[61]:


df["age"].describe()


# # plot the distribution of age variable

# In[65]:


f, ax = plt.subplots(figsize=(10,6))
x = df["age"]
ax = sns.distplot(x, bins=10)
plt.show()


# # analyse the age and target varible

# In[70]:


f, ax = plt.subplots(figsize=(8,6))
sns.stripplot(x="target", y="age", data=df)
plt.show()


# In[71]:


f, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x="target", y="age", data=df)
plt.show()


# # analyze age and trestbps variable

# In[73]:


s, ax = plt.subplots(figsize=(8,6))
ax = sns.scatterplot(x="age", y="trestbps", data=df)
plt.show()


# the above acatter plot shows that there is no correlation between age and trestbps variable

# In[74]:


f, ax = plt.subplots(figsize=(8,6))
sns.regplot(x="age", y="trestbps", data=df)
plt.show()


# In[75]:


f, ax = plt.subplots(figsize=(8,6))
ax = sns.scatterplot(x="age", y="chol", data=df)
plt.show()


# In[76]:


f, ax = plt.subplots(figsize=(8,6))
sns.regplot(x="age", y="chol", data=df)
plt.show()


# analyze chol and thalach variable

# In[78]:


f, ax = plt.subplots(figsize=(8,6))
ax = sns.scatterplot(x="chol", y="thalach", data=df)
plt.show()


# In[79]:


f, ax = plt.subplots(figsize=(8,6))
ax = sns.regplot(x="chol", y="thalach", data=df)
plt.show()


# In[80]:


#Dealing with missing value


# In[81]:


df.isnull().sum()


# In[82]:


# Check with Assert statement


# In[85]:


assert pd.notnull(df).all().all()


# In[86]:


assert (df >= 0).all().all()


# # outlier detection 

#  i will make a boxplot to visualise outlier in the continous numerical variables

# In[90]:


df["age"].describe()


# # box plot of age variable

# In[91]:


f, ax= plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["age"])
plt.show


# In[92]:


df["trestbps"].describe()


# # box-plot of trestbps variable

# In[94]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["trestbps"])
plt.show()


# # chol variable

# In[95]:


df["chol"].describe()


# # Boxplot of chol variables

# In[97]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["chol"])
plt.show()


# # thalach variable

# In[98]:


df["thalach"].describe()


# # boxplot of thalach variable

# In[101]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["thalach"])
plt.show()


#  # oldpeak variable

# In[104]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["oldpeak"])
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[144]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


# In[145]:


data = pd.read_csv('/Users/guglielmolatorre/Desktop/Kaggle diabetes/diabetes_prediction_dataset.csv')


# In[146]:


data.head()


# In[147]:


data.isna().sum()


# In[148]:


data.smoking_history.unique()


# In[149]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
transformed = ohe.fit_transform(data[['smoking_history']])
data[ohe.categories_[0]] = transformed.toarray()

data.head()


# In[150]:


#change datatype from float to int
data.rename(columns={'No info' : 'no_info'})


# In[151]:


X = data.iloc[:,:-1].values
y = data.iloc[:,-7].values


# In[152]:


data['gender'] = data['gender'].replace('Female', '0')
data['gender'] = data['gender'].replace('Male', '1')
data['gender'] = data['gender'].replace('Other', '2')

data.head()


# In[153]:


data_less = data.iloc[:, [0,1,3,4,5,6,7,8]]
data_less.head()

data['gender'] = data.gender.astype(int)

data_less = data_less.drop('smoking_history', axis = 1)

data_less.head()


# In[180]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

# LOGISTIC REGRESSION

#BMI predictor

X = data_less['bmi'].values
y = data_less['diabetes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

lr = LogisticRegression()

# need to reshape the data as .fit expects a 1-1 matrix rather thana single value
lr.fit(X_train.reshape(-1,1), y_train)

y_pred = lr.predict(X_test.reshape(-1,1))

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)


# In[166]:


X = data_less['HbA1c_level'].values
y = data_less['diabetes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

lr = LogisticRegression()

# need to reshape the data as .fit expects a 1-1 matrix rather thana single value
lr.fit(X_train.reshape(-1,1), y_train)

y_pred = lr.predict(X_test.reshape(-1,1))

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)


# In[168]:


X = data_less['age'].values
y = data_less['diabetes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

lr = LogisticRegression()

# need to reshape the data as .fit expects a 1-1 matrix rather thana single value
lr.fit(X_train.reshape(-1,1), y_train)

y_pred = lr.predict(X_test.reshape(-1,1))

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)


# In[169]:


X = data_less['heart_disease'].values
y = data_less['diabetes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

lr = LogisticRegression()

# need to reshape the data as .fit expects a 1-1 matrix rather thana single value
lr.fit(X_train.reshape(-1,1), y_train)

y_pred = lr.predict(X_test.reshape(-1,1))

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)


# In[179]:


#KNN

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors = 2)
knc.fit(X_train.reshape(-1,1), y_train)

y_pred = knc.predict(X_test.reshape(-1,1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


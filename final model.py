#!/usr/bin/env python
# coding: utf-8

# # 1 Import Necessary Libraries

# In[ ]:





# In[ ]:





# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder


# # 2 Get Data

# In[72]:


Data = pd.read_csv("pronostico_dataset.csv", sep=";")
Data


# In[73]:


Data.shape


# In[ ]:





# In[74]:


Data.drop(columns='ID',inplace=True)


# In[75]:


# we use labelencoder on target variable 
label_encoder= LabelEncoder()
Data.iloc[:,-1]=label_encoder.fit_transform(Data.iloc[:,-1])


# In[76]:


#Age outlier replace by median value of the age feature

median=Data.loc[Data['age']<82,'age'].median()
median2=Data.loc[Data['age']>37,'age'].median()
Data.loc[Data.age>82,'age']=np.nan
Data.loc[Data.age<37,'age']=np.nan
Data.fillna(median,inplace=True)
Data.fillna(median2,inplace=True)


# In[77]:


#systolic_bp outlier replace by median value of the age feature

median=Data.loc[Data['systolic_bp']<128,'systolic_bp'].median()
median2=Data.loc[Data['systolic_bp']>73,'systolic_bp'].median()
Data.loc[Data.systolic_bp>128,'systolic_bp']=np.nan
Data.loc[Data.systolic_bp<73,'systolic_bp']=np.nan
Data.fillna(median,inplace=True)
Data.fillna(median2,inplace=True)


# In[78]:


#diastolic_bp outlier replace by median value of the age feature

median=Data.loc[Data['diastolic_bp']<115,'diastolic_bp'].median()
median2=Data.loc[Data['diastolic_bp']>65,'diastolic_bp'].median()
Data.loc[Data.diastolic_bp>115,'diastolic_bp']=np.nan
Data.loc[Data.diastolic_bp<65,'diastolic_bp']=np.nan
Data.fillna(median,inplace=True)
Data.fillna(median2,inplace=True)


# In[79]:


#cholesterol outlier replace by median value of the age feature

median=Data.loc[Data['cholesterol'] < 127, 'cholesterol'].median()
median2=Data.loc[Data['cholesterol'] > 73, 'cholesterol'].median()
Data.loc[Data.cholesterol>127, 'cholesterol'] = np.nan
Data.loc[Data.cholesterol<73, 'cholesterol'] = np.nan
Data.fillna(median, inplace=True)
Data.fillna(median2, inplace=True)


# In[80]:


# Splitting the independent and dependent variables
X = Data.drop(columns="prognosis", axis=1)
Y = Data.prognosis
X.shape, Y.shape


# ## FEATURE SCALING

# In[81]:


from sklearn.preprocessing import StandardScaler


# In[82]:


# standardization 
std_scaler = StandardScaler()
X_scaler = std_scaler.fit_transform(X) 
print(X_scaler)


# In[83]:


#Spliting the variable into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=2903)


# # MODEL BUILDING

# ## GradientBoostingClassifier

# In[84]:


from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                       learning_rate=0.1, loss='deviance', max_depth=3,
                                       max_features=None, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=100,
                                       n_iter_no_change=None,
                                       random_state=2903, subsample=1.0, tol=0.0001,
                                       validation_fraction=0.1, verbose=0,
                                       warm_start=False)
gbc_model.fit(X_train,y_train)


# ### Validation of model

# In[85]:


y_pred_gbc_train= gbc_model.predict(X_train)


# In[86]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[87]:


print("Accuracy of training model",accuracy_score(y_train,y_pred_gbc_train))
print("==============================================================================")
print("Confusion Matrics",confusion_matrix(y_train,y_pred_gbc_train))
print("==============================================================================")
print("Classificaton Report ",classification_report(y_train,y_pred_gbc_train))


# In[88]:


y_pred_gbc_test= gbc_model.predict(X_test)


# In[89]:


print("Accuracy of testing model",accuracy_score(y_test,y_pred_gbc_test))
print("==============================================================================")
print("Confusion Matrics",confusion_matrix(y_test,y_pred_gbc_test))
print("==============================================================================")
print("Classificaton Report",classification_report(y_test,y_pred_gbc_test))


# In[90]:


#Create pickel file using serialization
# import pickle
# pickle_out=open("gbc_model.pkl","wb") 
# pickle.dump(gbc_model,pickle_out)
# pickle_out.close()


# In[91]:


import pickle


# In[98]:


# with open('gbc_model_pickle','wb') as f:
#     pickle.dump(gbc_model,f)
#     f.close()


# In[99]:


pickle.dump(gbc_model, open('model.pkl', 'wb'))


# In[ ]:





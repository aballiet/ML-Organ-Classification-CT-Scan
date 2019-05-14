#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[2]:


import glob

path = r'C:\Users\antoi\Documents\PIR\PIR\dataset' # chemin du dataset
all_files = glob.glob(path + "/*.csv")

xtab = pd.DataFrame()
ytab = pd.DataFrame()

#on donne un label aux colonnes
cols = [i for i in range(1089)]

for filename in all_files:
    data = pd.read_csv(filename)
    
    df_x = data.iloc[:,1:]
    df_y = data.iloc[:,0]
    
    df_x.columns = cols
    df_y.columns = [0]
    
    xtab = pd.concat([xtab, df_x], sort=False, axis=0)
    ytab = pd.concat([ytab, df_y], sort=False, axis=0)


# In[3]:


xtab


# In[ ]:


ytab


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(xtab, ytab, test_size=0.1, random_state=4)


# In[ ]:


x_train.head()


# In[ ]:


rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)


# In[ ]:

s=y_test.values
pred=rf.predict(x_test)


# In[ ]:

count = 0 

for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1


# In[ ]:


print(count/len(pred))


# In[ ]:





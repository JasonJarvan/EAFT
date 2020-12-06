
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import re


# In[2]:

def analyse(csvfilename)
    data = pd.read_csv(csvfilename)#读取dataafterfilter.csv


    # In[3]:


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    count_vec = CountVectorizer()
    # cross validation
    #x1,y1=data.iloc[:3000,3].fillna(' '),data.iloc[:3000,1]
    #x2,y2=data.iloc[6001:,3].fillna(' '),data.iloc[6001:,3]
    #x_train,y_train=x1.append(x2),y1.append(y2)
    x_train, y_train = data.iloc[1:, 3].fillna(' ') , data.iloc[1:, 1]#train all the words after 3000，x_train  is content，y_train is class
    x_test, y_test = data.iloc[:1, 3].fillna(' '), data.iloc[:1, 1]#test first 3000 words
    # word count by CountVectorizer
    x_train_mnb = count_vec.fit_transform(x_train)
    x_test_mnb = count_vec.transform(x_test)


    # In[5]:


    from sklearn import svm  #use svm to classify.
    from sklearn.linear_model import SGDClassifier
    #predict by SVM
    dtc = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)

    dtc.fit(x_train_mnb, y_train)
    #print(dtc.predict(x_test_mnb))
    return dtc.predict(x_test_mnb)


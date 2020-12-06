
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import re


# In[2]:
def sAnalyse(test_data):#let ScoreClassifier be the sAnalyse method


    data = pd.read_csv("scoreallafterfilter.csv")#read dataafterfilter.csv


    # In[3]:


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    count_vec = CountVectorizer()
    # cross validation
    #x1,y1=data.iloc[:3000,3].fillna(' '),data.iloc[:3000,1]
    #x2,y2=data.iloc[6001:,3].fillna(' '),data.iloc[6001:,3]
    #x_train,y_train=x1.append(x2),y1.append(y2)
    x_train, y_train = data.iloc[1:, 0].fillna(' ') , data.iloc[1:, 1]#train all the words after 3000，x_train is content，y_train is class
    x_test = test_data.iloc[:1, 0].fillna(' ')
    # word count by CountVectorizer
    x_train_mnb = count_vec.fit_transform(x_train)
    x_test_mnb = count_vec.transform(x_test)


    # In[4]:


    from sklearn import svm
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import classification_report
    from sklearn.externals import joblib
    #predict by SVM
    dtc = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)

    dtc.fit(x_train_mnb, y_train)
    #print(dtc.predict(x_test_mnb))

    result = None
    try:
        result = dtc.predict(x_test_mnb)[0]
    except Exception as e:
        print(str(e))

    return result

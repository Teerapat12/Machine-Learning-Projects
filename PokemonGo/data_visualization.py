# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_excel("pokemon TW TH search.xlsx",skiprows=1)

def get_RT_user(comment):
    
    if("RT" in comment and "@" in comment and ":" in comment):
        
        return "@"+comment.split("@")[1].split(":")[0]
    else:
        return "None"
    
def compute_polar_score(comment):
    positive_polar_words = ["ดี","สุดยอด","เจ๋ง","สนุก","มันส์"]
    negative_polar_words = ["ห่วย","กาก","งง","เบื่อ","รำคาน","ไม่ชอบ","เปลี่ยนเกม","อะไรก็ไม่รู้","ควย","สัด","สัส","สาด","สาส"]
    #TOdo, Polar word หลาย level 
    score = 0
    for word in positive_polar_words:
        if(word in comment):
            score+=1
    for word in negative_polar_words:
        if(word in comment):
            score-=1
    return score
    
        


dataset['RT_user'] = dataset['Tweet Text'].apply(get_RT_user)
dataset['opinion'] = dataset['Tweet Text'].apply(compute_polar_score)

print(dataset['opinion'].value_counts())
pos = dataset[dataset['opinion']>0]
neg = dataset[dataset['opinion']<0]

vectorizer = CountVectorizer(analyzer = "char",   \
                             ngram_range=(4, 7),   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             max_features = 2000) 

vectorized_tweet = vectorizer.fit_transform(dataset['Tweet Text'])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectorized_tweet, dataset['opinion'], test_size = 0.2, random_state = 0)

# Fitting the SVR to the dataset  (Support Vector Regression)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))

mytest = ['เกมเหี้ยอะไรเนี่ยห่วยสัดหมาอย่างกากเลยโว้ยย','สุดยอดเลยครับ สนุกมากๆ เล่นแล้วติดเลย 555']
y_pred2 = regressor.predict(vectorizer.transform(mytest))

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#import seaborn as sns
#Import dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
breed_le = LabelEncoder()
breed_le.fit(train['Breed'].append(test['Breed']))
color_feature = ['Black', 'Blue', 'Brindle', 'Brown', 'Orange', 'Red', 'Tabby', 'Tan', 'Tricolor', 'White']
def convert_age_today(row):
    age_string = row['AgeuponOutcome']
    if(pd.isnull(age_string)):
        return np.nan
    [age,unit] = age_string.split(" ")
    unit = unit.lower()
    if("day" in unit):
        if age=='0': return 1
        return int(age)
    if("week" in unit):
        if(age)=='0': return 7
        return int(age)*7
    elif("month" in unit):
        if(age)=='0': return 30
        return int(age) * 4*7
    elif("year" in unit):
        if(age)=='0': return 365
        return int(age) * 4*12*7

def find_popular_cate(df,cate,threshold = 500,):
    allword = {}
    for index,row in df.iterrows():
        val = row[cate]
        infos = " ".join(val.split("/")).split(" ")
        for info in infos:
            if info in allword:
                allword[info]+=1
            else:
                allword[info] = 1
    selectedWord = []
    for key in sorted(allword):
        if(allword[key]>threshold):
            selectedWord.append(key)
    return selectedWord

def clean_animals(tdf):
    tdf['DateTime'] = pd.to_datetime(tdf['DateTime'])
    for index,row in tdf.iterrows():
        dateTime = row['DateTime']
        tdf.set_value(index,'Year',dateTime.year)
        tdf.set_value(index,'Month',dateTime.month)
        tdf.set_value(index,'Day',dateTime.day)
        tdf.set_value(index,'DayOfWeek',dateTime.dayofweek)
        tdf.set_value(index,'Hour',dateTime.hour)
        color = row['Color']
        #Seperate color by space so that we can use CountVectorizer
        tdf.set_value(index,'Color'," ".join(color.split("/")))
        #Check if pet is a mix or not
        if('mix' in row['Breed'].lower()):
            tdf.set_value(index,'IsMix',1)
        else:
            tdf.set_value(index,'IsMix',0)
        for color in color_feature:
            if color in row['Color']:
                tdf.set_value(index,color,1)
            else:
                tdf.set_value(index,color,0)
        
        if(row['AnimalType']=='Dog'):
            tdf.set_value(index,'AnimalType',1)
        if(row['AnimalType']=='Cat'):
            tdf.set_value(index,'AnimalType',0)
        
        gender = row['SexuponOutcome']
        if(gender=="Unknown" or pd.isnull(gender)):
            tdf.set_value(index,'Intact',-1)
            tdf.set_value(index,'isMale',-1)
        else:
            gender = gender.split(" ")
            if(gender[0]!="Intact"):
                tdf.set_value(index,'Intact',0)
            else:
                tdf.set_value(index,'Intact',1)
            if(gender[1]=="Male"):
                tdf.set_value(index,'isMale',1)
            else:
                tdf.set_value(index,'isMale',0)        
    tdf['AgeInDays'] = tdf.apply(convert_age_today, axis=1)
    tdf['AgeInDays'] = train['AgeInDays'].fillna(train['AgeInDays'].median())
    tdf['Breed'] = breed_le.transform(tdf['Breed'])
    return tdf

selectCol = ['AnimalType', 'Breed', 
       'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'IsMix', 'Black', 
       'Brindle', 'Orange', 'Red', 'Tabby',  'Tricolor',
        'Intact', 'isMale', 'AgeInDays']
target = ['OutcomeType']
train = clean_animals(train)
test = clean_animals(test)
X_train = train[selectCol]
X_test = test[selectCol]
y_train = train[target]
#Encode the output
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#y_train = y_train.values.ravel()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y = classifier.predict(X_train)

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = y,exog = X_train).fit()
regressor_OLS.summary()


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

ID = test['ID']

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

columns = ['ID','Adoption','Died','Euthanasia','Return_to_owner','Transfer']
predict_df = pd.DataFrame(
    {'ID': ID,
     'Adoption': y_prob[:,0],
     'Died': y_prob[:,1],
     'Euthanasia': y_prob[:,2],
     'Return_to_owner': y_prob[:,3],
     'Transfer': y_prob[:,4]
     },  columns=columns)
predict_df.to_csv('animals_outcome_prediction_v2.csv',index=False)
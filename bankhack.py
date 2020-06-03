# -*- coding: utf-8 -*-
"""BankHack.ipynb


**Machine Learning for Banking||Interest rate prediction**

*goal is to use a training dataset to predict the loan rate category (1 / 2 / 3) that will be assigned to each loan in our test set.*
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost

train = pd.read_csv('train_jh.csv')
test = pd.read_csv('test_jh.csv')
train.head(10)

train['Months_Since_Deliquency'].mean()

train['Length_Employed'].value_counts()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train[['Length_Employed','Home_Owner']] = imputer.fit_transform(train[['Length_Employed','Home_Owner']])
test[['Length_Employed', 'Home_Owner']] = imputer.fit_transform(test[['Length_Employed', 'Home_Owner']])

imputer_2 = SimpleImputer(missing_values=np.nan, strategy='mean')
train[['Annual_Income']] = imputer_2.fit_transform(train[['Annual_Income']])
test[['Annual_Income']] = imputer_2.fit_transform(test[['Annual_Income']])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Length_Employed'] = train['Length_Employed'].astype(str)
test['Length_Employed'] = test['Length_Employed'].astype(str)
train['Length_Employed'] = le.fit_transform(train['Length_Employed'])
test['Length_Employed'] = le.fit_transform(test['Length_Employed'])



train['Home_Owner'] = train['Home_Owner'].astype(str)
test['Home_Owner'] = test['Home_Owner'].astype(str)
train['Home_Owner'] = le.fit_transform(train['Home_Owner'])
test['Home_Owner'] = le.fit_transform(test['Home_Owner'])


train['Income_Verified'] = train['Income_Verified'].astype(str)
test['Income_Verified'] = test['Income_Verified'].astype(str)
train['Income_Verified'] = le.fit_transform(train['Income_Verified'])
test['Income_Verified'] = le.fit_transform(test['Income_Verified'])


train['Gender'] = train['Gender'].astype(str)
test['Gender'] = test['Gender'].astype(str)
train['Gender'] = le.fit_transform(train['Gender'])
test['Gender'] = le.fit_transform(test['Gender'])


train['Purpose_Of_Loan'] = train['Purpose_Of_Loan'].astype(str)
test['Purpose_Of_Loan'] = test['Purpose_Of_Loan'].astype(str)
train['Purpose_Of_Loan'] = le.fit_transform(train['Purpose_Of_Loan'])
test['Purpose_Of_Loan'] = le.fit_transform(test['Purpose_Of_Loan'])
train.head(10)

def clean_currency(x):
   
    if isinstance(x, str):
        return(x.replace(',', ''))
    return(x)

train['Loan_Amount_Requested'] = train['Loan_Amount_Requested'].apply(clean_currency).astype('float')
test['Loan_Amount_Requested'] = test['Loan_Amount_Requested'].apply(clean_currency).astype('float')

train.info()

train['Length_Employed'].value_counts()

train[['Length_Employed']] = train[['Length_Employed']].replace(10,0.5)
train[['Length_Employed']] = train[['Length_Employed']].replace(1,10)
train[['Length_Employed']] = train[['Length_Employed']].replace(0,1)
test[['Length_Employed']] = test[['Length_Employed']].replace(10,0.5)
test[['Length_Employed']] = test[['Length_Employed']].replace(1,10)
test[['Length_Employed']] = test[['Length_Employed']].replace(0,1)
train['Length_Employed'].value_counts()

train.head(10)

df = train[['Months_Since_Deliquency']]
check = df.isnull()
check.rename(columns={'Months_Since_Deliquency':'Indicator'}, inplace=True)
check['Indicator'] = check['Indicator'].astype(int)
train2 = pd.concat([train, check], axis=1, sort=False)

train2.head()

df = test[['Months_Since_Deliquency']]
check2 = df.isnull()
check2.rename(columns={'Months_Since_Deliquency':'Indicator'}, inplace=True)
check2['Indicator'] = check2['Indicator'].astype(int)
test2 = pd.concat([test, check2], axis=1, sort=False)

test2.head()

train2['Months_Since_Deliquency'].mean()

test2['Months_Since_Deliquency'].mean()

train2['Months_Since_Deliquency'] = train2['Months_Since_Deliquency'].fillna(34)
test2['Months_Since_Deliquency']= test2['Months_Since_Deliquency'].fillna(34)

train2.isnull().sum()

test2.isnull().sum()

"""train2.head()"""

from xgboost import XGBClassifier
model = XGBClassifier()

y = train2['Interest_Rate']
features = ['Loan_Amount_Requested','Length_Employed','Income_Verified','Purpose_Of_Loan','Debt_To_Income','Inquiries_Last_6Mo','Months_Since_Deliquency','Number_Open_Accounts','Total_Accounts','Gender','Indicator']
x = train2[features]

model.fit(x,y)

Xtest = test2[features]

predictions = model.predict(Xtest)
output = pd.DataFrame({'Loan_ID': test2.Loan_ID, 'Interest_Rate': predictions})

output.to_csv('#Location')

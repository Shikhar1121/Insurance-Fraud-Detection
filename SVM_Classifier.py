# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:55:23 2021

@author: Shikhar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv("Data/insuranceFraud.csv")

df.head()

df.shape

# in the dataset the missing values are reported by 
# ? so we replace them with nan values
df = df.replace('?',np.nan)

df.columns

#some columns re not necessry for prediction so we drop them
cols_to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_date','incident_state', 'incident_city','incident_location','auto_make','auto_model','insured_hobbies']


#dropping the columns
df.drop(columns= cols_to_drop,inplace=True)

#checking for missing data
df.isnull().sum()


#as the missing values are categorical values we'll use categorical imputer
from sklearn_pandas import CategoricalImputer  

imputer = CategoricalImputer()

#imputing missin values
df['collision_type'] = imputer.fit_transform(df['collision_type'])
df['property_damage'] = imputer.fit_transform(df['property_damage'])
df['police_report_available'] = imputer.fit_transform(df['police_report_available'])


#segregating categorical features for encoding
cat_df = df.select_dtypes(include = ['object']).copy()


# custom mapping for encoding
cat_df['policy_csl'] = cat_df['policy_csl'].map({'100/300' : 1, '250/500' : 2.5 ,'500/1000':5})
cat_df['insured_education_level'] = cat_df['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
cat_df['incident_severity'] = cat_df['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
cat_df['insured_sex'] = cat_df['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
cat_df['property_damage'] = cat_df['property_damage'].map({'NO' : 0, 'YES' : 1})
cat_df['police_report_available'] = cat_df['police_report_available'].map({'NO' : 0, 'YES' : 1})
cat_df['fraud_reported'] = cat_df['fraud_reported'].map({'N' : 0, 'Y' : 1})

# auto encoding of categorical variables
for col in cat_df.drop(columns=['policy_csl','insured_education_level','incident_severity','insured_sex','property_damage','police_report_available','fraud_reported']).columns:
    cat_df= pd.get_dummies(cat_df, columns=[col], prefix = [col], drop_first=True)
    
cat_df.head()


# extracting the numerical columns
num_df = df.select_dtypes(include=['int64']).copy()


# combining the Numerical and categorical dataframes to get the final dataset
final_df=pd.concat([num_df,cat_df], axis=1)


#separating dependent and independent features
x = final_df.drop('fraud_reported',axis = 1)
y = final_df['fraud_reported']

#finding correlation
num_df.corr()




#dropping age because age is not affecting the insurance fraud and total claim amount is the summation of 3 columns
x.drop(columns=['age','total_claim_amount'], inplace=True)


# splitting the data into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=355 )

num_df=x_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]



#scaling numericalnumerical features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaled_data=scaler.fit_transform(num_df)
scaled_num_df= pd.DataFrame(data=scaled_data, columns=num_df.columns,index=x_train.index)
scaled_num_df.shape

scaled_num_df.isnull().sum()

x_train.drop(columns=scaled_num_df.columns, inplace=True)


x_train=pd.concat([scaled_num_df,x_train],axis=1)



#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
sv_classifier=SVC()


#prediction
y_pred = sv_classifier.fit(x_train,y_train).predict(x_test)

from sklearn.metrics import accuracy_score
sc=accuracy_score(y_test,y_pred)

from sklearn.model_selection import GridSearchCV
param_grid = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
             "C":[0.1,0.5,1.0,2.0,2.5,3.0],
             "random_state":[0,100,200,300]}

grid = GridSearchCV(estimator=sv_classifier, param_grid=param_grid, cv=5,  verbose=3)

grid.fit(x_train,y_train)


grid.best_estimator_

sv_classifier2=SVC(C=0.1, kernel='linear', random_state=0)

y_pred = sv_classifier2.fit(x_train,y_train).predict(x_test)

from sklearn.metrics import accuracy_score
sc2=accuracy_score(y_test,y_pred)
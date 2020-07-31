# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:00:12 2019

@author: ai
"""

# classifying personal income#
#required packages#

#to work with dataframe#
import pandas as pd

#to work with numerical operation#
import numpy as np

#to visualize data#
import seaborn as sns

#to partition the data#
from sklearn.model_selection import train_test_split

#importing library for logistic regression#
from sklearn.linear_model import LogisticRegression

#importing performance metrics - accuracy score and confusion matrix#
from sklearn.metrics import accuracy_score,confusion_matrix

##############################################################
#importing data
data_income=pd.read_csv('income.csv')


#creating data copy of original data

data=data_income.copy(deep=True)

'''''
exploratory data analysis

#1.getting to know data
#2. data preprocessing (missing values)
#3.Crosstables and data visualization

'''''
#Getting to know data
#============================================
#*******To Check variables data type

print(data.info())

#checking for missing values

data.isnull()
print('data columns with null values:\n',data.isnull().sum())

#********No missing values

#********summary of numerical and categorical vairables
summary_num=data.describe(include=['int64','object'])
print(summary_num)

#********Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()
 
#********Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
 
''''
go back and read the data by including "na_values=['?']" to csv command
 
''''

data=pd.read_csv('income.csv',na_values=[' ?'])

#**********
#data preprocessing
#************

data.isnull().sum()

missing=data[data.isnull().any(axis=1)]
#axis=1 => to consider at least one column value is missing

''''points to consider
1. Missing values in jobtype=1806
2. missing values in occupation=1816
3. There are 1809 rows where two specific columns i.e. occupation & job type have missing values
4. (1816-1806=7)  => you still have occupation unfilled for these 7 rows.Because , jobtype is Never Worked
''''

data.dropna(axis=0,inplace=True)

#******** relationship between independent variable

correlation=data.corr()

#*******************
#cross tables & Data visualization
#***********************
#Extracting the columns names

data.columns

#=======================================

#Gender proportion table:

#===================================

gender=pd.crosstab(index=data['gender'],columns='count',normalize=True)
print(gender)

#=========================

#Gender vs salary status:

#=========================

gender_salstat=pd.crosstab(index=data['gender'],columns=data['SalStat'],margins=True,normalize='index')
print(gender_salstat)

#=========================================================

#Frequency distribution of "Salary Status"

#==========================================================

SalStat = sns.countplot(data['SalStat'])

'''
75% of people's salary status is <=50,000
25% of people's salary status is >=50,000
'''
 ##############  histogram of age ##############################
 
 sns.distplot(data['age'],bins=10,kde=False)
#  People with age 20-45 age are high  in Frequency
 
 ########################## box-plot - age vs SalStat  ################
 
 sns.boxplot(x=data['SalStat'],y=data['age'],data=data)
 data.groupby('SalStat')['age'].median()
 
 ## people with 35-50 age are more likely to earn >50000 usd pa
 ## people with 25-45 age are more likely to earn <50000 uda pa
 
 ############ bar graph jobtype hue salstat     ############
 
 sns.countplot(y='JobType',data=data,hue='SalStat')
 pd.crosstab(index=data['JobType'],columns=data['SalStat'],margins=True,normalize='index')
 '''
 56% people who are self employed earing > 50000 usd pa
 
 '''
 
 ############# bar graph Edtype hue salstat    ################
 sns.countplot(y='EdType',data=data,hue='SalStat')
 pd.crosstab(index=data['EdType'],columns=data['SalStat'],margins=True,normalize='index')
'''
doctrates ,masters, prof-school are more likely to earn >50000 usd pa

'''

 ################ bar graph occupation vs salstat   ###########
 sns.countplot(y='occupation',data=data,hue='SalStat')
 pd.crosstab(index=data['occupation'],columns=data['SalStat'],margins=True,normalize='index')

'''
prof speciality and exec manager are more likely to earn >50000 uds pa
'''

#################################################################################################
#################  logistic Regeression   ##########################################################
##################################################################################################

data2=data.copy(deep=True)

#reindexing the salary status names to 0 and 1

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':'0',' greater than 50,000':'1'})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

# check for columns name it changes the columns name also

# Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

# Separating the columns names
features=list(set(columns_list)-set(['SalStat_1']))
print(features)

# storing the output values in y
y=new_data['SalStat_1'].values
print(y)

# storing the values from input features
x=new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

# Make an intances of the Model
logistic=LogisticRegression()

# Fitting the values of x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_


# Prediction from test data
prediction=logistic.predict(test_x)
print(prediction)

# confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

# calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


# Printing the misclassified values from prediction

print('Misclassified samples: %d'%(test_y!=prediction).sum())

###################################################################
######  Logistic Regression _removing insignificant variables
# do it with all previous defined variable deleted
#########################################################################

data2=data.copy(deep=True) 

#reindexing the salary status names to 0 and 1

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':'0',' greater than 50,000':'1'})
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

# Separating the columns names
features=list(set(columns_list)-set(['SalStat_1']))
print(features)

# storing the output values in y
y=new_data['SalStat_1'].values
print(y)

# storing the values from input features
x=new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

# Make an intances of the Model
logistic=LogisticRegression()

# Fitting the values of x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_


# Prediction from test data
prediction=logistic.predict(test_x)
print(prediction)

# confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

# calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


# Printing the misclassified values from prediction

print('Misclassified samples: %d'%(test_y!=prediction).sum())

# ===================================================================

#  KNN
# delete all user defined variables before applying below code


#=======================================================================

# importing the library of knn
from sklearn.neighbors import KNeighborsClassifier

#import library for plotting

import matplotlib.pyplot as plt

# storing the K Nearest Neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

# fitting the values for x and y
KNN_classifier.fit(train_x,train_y)

#prdicting the test values with model
prediction=KNN_classifier.predict(test_x)

# Performance metric check
confusion_matrix=confusion_matrix(test_y,prediction)

print('\t','Predicted values')
print('original values','\n',confusion_matrix)

# calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


# Printing the misclassified values from prediction

print('Misclassified samples: %d'%(test_y!=prediction).sum())

''''
effect of k on classifier

''''

Misclassified_sample=[]
# Calculating error for k values between 1 and 20
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=pred_i).sum())
    
print(Misclassified_sample)

##########################################################################
######   end of script   #################################################
''''
we can check for further improvement by removing 
more insignificant features in logistic case as 
well as knn case\\

''''
   






















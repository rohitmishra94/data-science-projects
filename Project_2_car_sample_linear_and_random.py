# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:59:24 2019

@author: ai
"""

#==========================================================
# Predicting Price of Pre Owned Cars
#============================================================
import numpy as np
import pandas as pd
import seaborn as sns


#=============================================================
#setting dimensions for plot
#==============================================================

sns.set(rc={'figure.figsize' :(11.7,8.27)})


#=========================================================
# Reloading CSV file
#================================================================

cars_data=pd.read_csv('cars_sampled.csv')

#==============================================================
#creating copy
#===========================================================
cars=cars_data.copy()

#================================================================
# structure of dataset
#============================================================
 cars.info()

#===========================================================
# Summarizing data
#=======================================================

 cars.describe()
 pd.set_option('display.float_format', lambda x: '%.3f' % x)
 cars.describe()

 # To display maximum set of columns
 pd.set_option('display.max_columns',500)
 cars.describe()

 #======================================================================
 # Dropping unwanted columns
 #=========================================================================

 col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
 cars=cars.drop(columns=col, axis=1)

 #============================================================
 # Removing Duplicates Records
 #=============================================================

 cars.drop_duplicates(keep='first',inplace=True)
 #470 duplicates drop

 #==============================================================
 # Data cleaning
 #==============================================================

 # No. of missing Values in each column
 cars.isnull().sum()

 #Variable yearOfRegistration
 yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
 sum(cars['yearOfRegistration']>2018)
 sum(cars['yearOfRegistration']<1950)
 sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars)

 # working range - 1950 and 2018

 # Variable Price
 price_count=cars['price'].value_counts().sort_index()
 sns.distplot(cars['price'])
 cars['price'].describe()
 sns.boxplot(y=cars['price'])
 sum(cars['price']>150000)
 sum(cars['price']<100)

 # working range - 100 and 150000


 # variable PowerPS
 power_count=cars['powerPS'].value_counts().sort_index()
 sns.distplot(cars['powerPS'])
 cars['powerPS'].describe()
 sns.boxplot(y=cars['powerPS'])
 sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)
 sum(cars['powerPS']>500)
 sum(cars['powerPS']<10)

# working range 10 and 500



 #============================================================================
 # working range of data
 #==============================================================================

 # working range of data

 cars=cars[(cars.yearOfRegistration <=2018) & (cars.yearOfRegistration >=1950)& (cars.price >=100)& (cars.price <= 150000)& (cars.powerPS >= 10)& (cars.powerPS <= 500)]

 # ~6700 records are dropped

 # Further to simplify - varible reduction
 # combining yearOfRegistration and monthOfRegistration

 cars['monthOfRegistration']/=12

 # creating new varible Age by adding yearOfRegistration and monthOfRegistration
 cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
 cars['Age']=round(cars['Age'],2)
 cars['Age'].describe()

 # Dropping year of Registration and month of registration
 cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)



 # visualizing parameters

 # Age
 sns.distplot(cars['Age'])
 sns.boxplot(y=cars['Age'])

 # price
 sns.distplot(cars['price'])
 sns.boxplot(y=cars['price'])

 #powerPS
 sns.distplot(cars['powerPS'])
 sns.boxplot(y=cars['powerPS'])


 # Visualizing parameters after narrowing working range
 # Age vs price
 sns.regplot(x='Age', y='price', scatter=True, fit_reg=False, data=cars)

 # cars priced higher are newer
 # with increase in age, price decrease
 # however some cars are priced higher with increase in age


 # powerPS vs price
 sns.regplot(x='powerPS',y='price',scatter=True, fit_reg=False, data=cars)


 # variable seller
 cars['seller'].value_counts()
 pd.crosstab(cars['seller'],columns='count',normalize=True)
 sns.countplot(x='seller',data=cars)

 #fewer cars have 'commercial'=> insignificant



 # Variable offerType
 cars['offerType'].value_counts()
 sns.countplot(x='offerType',data=cars)
 # All cars have 'Offer'=> Insignificant

 # Varible abtest
 cars['abtest'].value_counts()
 pd.crosstab(cars['abtest'],columns='count',normalize=True)
 sns.countplot(x='abtest',data=cars)

 #Equally distributed
 sns.boxplot(x='abtest',y='price',data=cars)

 # For every price value there is almost 50-50 distribution
 # Does not affect price => Insignificant

 # Variable vehicleType
 cars['vehicleType'].value_counts()
 pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
 sns.countplot(x='vehicleType',data=cars)
 sns.boxplot(x='vehicleType', y='price',data=cars)

 # 8 types limousine, small cars and station wagons max freq
 # vehicleType affects price

 # variable gearbox
 cars['gearbox'].value_counts()
 pd.crosstab(cars['gearbox'],columns='count',normalize=True)
 sns.countplot(x='gearbox', data=cars)
 sns.boxplot(x='gearbox', y='price',data=cars)

 # gearbox affects price

 # Variable model
 cars['model'].value_counts()
 pd.crosstab(cars['model'],columns='count',normalize=True)
 sns.countplot(x='model',data=cars)
 sns.boxplot(x='model',y='price',data=cars)

 #cars are distributed over many models
 # considered in modelling

 # variable kilometer
 cars['kilometer'].value_counts().sort_index()
 pd.crosstab(cars['kilometer'],columns='counts',normalize=True)
 sns.boxplot(x='kilometer',y='price',data=cars)
 cars['kilometer'].describe()
 sns.distplot(cars['kilometer'],bins=8,kde=False)
 sns.regplot(x='kilometer',y='price',scatter=True, fit_reg=False, data=cars)

 # considered in modelling

 # varible fuelType
 cars['fuelType'].value_counts()
 pd.crosstab(cars['fuelType'],columns='count',normalize=True)
 sns.countplot(x='fuelType',data=cars)
 sns.boxplot(x='fuelType',y='price',data=cars)

 # fuelType affects price

 # varible brand
 cars['brand'].value_counts()
 pd.crosstab(cars['brand'],columns='count',normalize=True)
 sns.countplot(x='brand',data=cars)
 sns.boxplot(x='brand',y='price',data=cars)

 #cars are distributed over many brands
 #cosidered for modelling

 #variable notRepairedDamage
 #yes- car is damaged but not rectified
 #no- car was damaged but has been rectified
 cars['notRepairedDamage'].value_counts()
 pd.crosstab(cars['notRepairedDamage'],columns='counts',normalize=True)
 sns.countplot(x='notRepairedDamage',data=cars)
 sns.boxplot(x='notRepairedDamage',y='price',data=cars)

 # AS expected the cars that require the damage to be repaired
 # fall under lower price ranges


 #=====================================================
 # Removing insignificant variables
 #==========================================================

 col=['seller','offerType','abtest']
 cars=cars.drop(columns=col,axis=1)
 cars_copy=cars.copy()

 #===================================================================
 # Correlation
 #======================================================================
  cars_select1=cars.select_dtypes(exclude=[object])
  correlation=cars_select1.corr()
  round(correlation,3)
  cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

  #===============================================================================


  '''
  we are going to build a linear regression and random forest model
  on two sets of data.
  1. data obtained by omitting rows with any missing value
  2. data obtained by imputing the missing value
  '''

  #========================================================================
  #omitting missing values
  #========================================================================

  cars_omit=cars.dropna(axis=0)

  # Converting categorical variables to dummy variables
  cars_omit=pd.get_dummies(cars_omit,drop_first=True)
  
  # check for drop_first which category dropped


 #==================================================================================
 # importing necessary Libraries
 #=================================================================================

 from sklearn.model_selection import train_test_split
 from sklearn.linear_model import LinearRegression
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.metrics import mean_squared_error

 #==================================================================================
 # model building with omitted data
 #====================================================================================

 # Separating input and output features
 x1 = cars_omit.drop(['price'],axis=1,inplace=False)
 y1=cars_omit['price']

 # plotting the variable price
 prices=pd.DataFrame({'1.before':y1,'2.after': np.log(y1)})
 prices.hist()

 # Transforming price as a logarithmic value
 y1=np.log(y1)

 #splitting data into train and test
 train_x,test_x,train_y,test_y=train_test_split(x1,y1,test_size=0.3,random_state=3)
 print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)


#=============================================================================
# baseline model for omitted data
#======================================================================

 ''''
 we are making a base model by using test data mean value
 this is to set a benchmark and to compare with our regression model

 ''''

# finding the mean for test data value
base_pred = np.mean(test_y)
print(base_pred)

# Repeating same value till length of test data
 base_pred=np.repeat(base_pred,len(test_y))

 # finding the RMSE
base_root_mean_squared_error=np.sqrt(mean_squared_error(test_y,base_pred))

print(base_root_mean_squared_error) # baseline model

#=====================================================================================
# linear Regression
#====================================================================================


# Setting intercept as True
lgr=LinearRegression(fit_intercept=True)

#Model
model_lin1=lgr.fit(train_x,train_y)

#predicting Model on test data
cars_predictions_lin1=lgr.predict(test_x)

# computing MSE ans RMSE
lin_mse1=mean_squared_error(test_y,cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1) # root mean squred erro for omitted data

# R squared Value
r2_lin_test1=model_lin1.score(test_x,test_y)
r2_lin_train1=model_lin1.score(train_x,train_y)
print(r2_lin_test1,r2_lin_train1) # r^2 values for test, train for omitted data

# regression diagnosis _ residual Plot analysis
residuals1=test_y-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residuals1,scatter=True, fit_reg=False, data=cars)
 residuals1.describe()

 #================================================================================================
 # random forest with omitted data
 #====================================================================================================

 #model parameters
 rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

 # Model
 model_rf1=rf.fit(train_x,train_y)
 
 # it will take some time in normal system

 # predicting model on test set
 cars_predictions_rf1=rf.predict(test_x)

 # computing mse and rmse
 rf_mse1=mean_squared_error(test_y,cars_predictions_rf1)
 rf_rmse1=np.sqrt(rf_mse1)
 print(rf_rmse1) # rmse in random forest model for omitted data


# R squared value
 r2_rf_test1=model_rf1.score(test_x,test_y)
 r2_rf_train1=model_rf1.score(train_x,train_y)
 print(r2_rf_test1,r2_rf_train1) # r^2 for test and train for omitted data in random forest

#======================================================================================
 # model building with imputed data
 #===================================================================================
 
 cars_imputed=cars.apply(lambda x: x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
 
 cars_imputed.isnull().sum()
 
 # converting categorical variables to dummy variables
 cars_imputed=pd.get_dummies(cars_imputed, drop_first=True)
 
 
 #=============================================================================================
 # model with imputed data
 #==========================================================================================================
 
 
# Separating input and output features
 x2 = cars_imputed.drop(['price'],axis=1,inplace=False)
 y2=cars_imputed['price']

 # plotting the variable price
 prices=pd.dataframe({'1.before':y2,'2.after': np.log(y2)})
 prices.hist()

 # Transforming price as a logarithmic value
 y2=np.log(y2)

 #splitting data into train and test
 train_x1,test_x1,train_y1,test_y1=train_test_split(x2,y2,test_size=0.3,random_state=3)
 print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)


#==================================================================================================
# baseline model for imputed data
#===========================================================================================================
 
 ''''
 we are making a base model by using test data mean value
 this is to set a benchmark and to compare with our regression model

 ''''

# finding the mean for test data value
base_pred = np.mean(test_y1)
print(base_pred)

# Repeating same value till length of test data
 base_pred=np.repeat(base_pred,len(test_y1))

 # finding the RMSE
base_root_mean_squared_error_imputed=np.sqrt(mean_squared_error(test_y1,base_pred))

print(base_root_mean_squared_error_imputed)

#===============================================================================================
# Linear Regression with imputed data
#=============================================================================================================

# Setting intercept as True
lgr2=LinearRegression(fit_intercept=True)

#Model
model_lin2=lgr2.fit(train_x1,train_y1)

#predicting Model on test data
cars_predictions_lin2=lgr2.predict(test_x1)

# computing MSE ans RMSE
lin_mse2=mean_squared_error(test_y1,cars_predictions_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

# R squared Value
r2_lin_test2=model_lin2.score(test_x1,test_y1)
r2_lin_train2=model_lin2.score(train_x1,train_y1)
print(r2_lin_test2,r2_lin_train2)

#============================================================================================================================
# Random forest with imputed data
#========================================================================================================================================

# model parameters
 rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

 # Model
 model_rf2=rf2.fit(train_x1,train_y1)

 # predicting model on test set
 cars_predictions_rf2=rf2.predict(test_x1)

 # computing mse and rmse
 rf_mse2=mean_squared_error(test_y1,cars_predictions_rf2)
 rf_rmse2=np.sqrt(rf_mse2)
 print(rf_rmse2)


# R squared value
 r2_rf_test2=model_rf2.score(test_x1,test_y1)
 r2_rf_train2=model_rf2.score(train_x1,train_y1)
  print(r2_rf_test2,r2_rf_train2)


####################################################################################################################################################
 
print(base_root_mean_squared_error) # baseline model
print(lin_rmse1) # root mean squred erro for omitted data
print(r2_lin_test1,r2_lin_train1) # r^2 values for test, train for omitted data
print(rf_rmse1) # rmse in random forest model for omitted data
print(r2_rf_test1,r2_rf_train1) # r^2 for test and train for omitted data in random forest
 
print(base_root_mean_squared_error_imputed)
print(lin_rmse2)
print(r2_lin_test2,r2_lin_train2)
print(rf_rmse2)
print(r2_rf_test2,r2_rf_train2)

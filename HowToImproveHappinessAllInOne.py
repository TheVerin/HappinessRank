#WhatMakeUsHappy?

#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#Importing dataset
RawData = pd.read_csv('2015.csv')

#First view of data
RawData.head()
RawData.dtypes
RawData.describe()  #there is no missing data :D
RawData.hist(bins = 50)
RawData.skew()

for col in RawData.select_dtypes(['object']):
    print(RawData[col].value_counts())

#Chcanging the names of columns
RawData.columns = ['Country', 'Region', 'Rank', 'Score', 'StandardError', 'Economy', 'Family', 
                   'Health', 'Freedom', 'Trust', 'Generosity', 'DystopiaResidual']

#Rounding data to 0.001
for col in RawData.select_dtypes(['int64', 'float64']):
    RawData.round(3)

#Visualization of data
sns.boxplot(data = RawData, x = 'Region', y = 'Score')
    
sns.stripplot(data = RawData, x = 'Economy', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'Family', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'Health', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'Freedom', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'Trust', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'Generosity', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'DystopiaResidual', y = 'Score', hue = 'Region')
sns.stripplot(data = RawData, x = 'StandardError', y = 'Score', hue = 'Region')

sns.violinplot(data = RawData, x = 'Region', y = 'Economy')
sns.violinplot(data = RawData, x = 'Region', y = 'Family')
sns.violinplot(data = RawData, x = 'Region', y = 'Health')
sns.violinplot(data = RawData, x = 'Region', y = 'Freedom')
sns.violinplot(data = RawData, x = 'Region', y = 'Trust')
sns.violinplot(data = RawData, x = 'Region', y = 'Generosity')
sns.violinplot(data = RawData, x = 'Region', y = 'DystopiaResidual')


#Preparing the dataset
#Choosing right variables
HappyData = RawData
def Drop():
    global HappyData
    HappyData.drop(['Country', 'Rank'], axis = 1, inplace = True)
    return HappyData
HappyData = Drop()

#Change Region into dummies
def RegionDummies():
    global HappyData
    Region_dummies = pd.get_dummies(HappyData['Region'], prefix = 'Region')
    HappyData = pd.concat([HappyData, Region_dummies], axis = 1,)
    HappyData.drop('Region', axis = 1, inplace = True)
    HappyData = HappyData.iloc[:, :-1]
    return HappyData
HappyData = RegionDummies()

#Spliting into independent (X) and dependent (Y) variables
X = HappyData.iloc[:, 1:]
Y = HappyData.iloc[:, 0]

#Scalling
from sklearn.preprocessing import StandardScaler
scaler_1 = StandardScaler()
X = scaler_1.fit_transform(X)

#Return to DataFrame
X = pd.DataFrame(X)
X.columns = ['StandardError', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 
             'DystopiaResidual', 'AustraliaAndNewZeland', 'CentralAndEsternEurope', 'EsternAsia',
             'LatinAmericaAndCaraibbean', 'MiddleEastAndNorthAfrica', 'NorthAmerica', 'SouthEastAsia',
             'SouthernAsia', 'Sub-SaharanAfrica']


#Spliting into train/test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#First Fitting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

LinReg = LinearRegression()
GBReg = GradientBoostingRegressor()

LinReg.fit(X_train, Y_train)
GBReg.fit(X_train, Y_train)
LinReg.coef_

#First Predictions
LinRegPred = LinReg.predict(X_train)
GBRegPred = GBReg.predict(X_train)

#Checking the RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

LinRegScore = sqrt(mean_squared_error(Y_train, LinRegPred))
GBRegScore = sqrt(mean_squared_error(Y_train, GBRegPred))

#View into results
print('Linear Regression score: ',LinRegScore)
print('Gradient Boosting score: ',GBRegScore)

#Taking the best alhorithm and optimalize
#Choosing the most important variables
from sklearn.feature_selection import RFE
from sklearn.cross_validation import cross_val_score
nrse_all = []
for m in np.arange(0,17):
    nrse_loop =[]
    for n in np.arange(5, 17, 1):
        selector = RFE(LinReg, n, 1)
        cv = cross_val_score(LinReg, X_train.iloc[:, selector.fit(X_train, Y_train).support_], Y_train, cv = 10, scoring = 'neg_mean_squared_error')
        nrse_loop.append(cv.mean())
    nrse_all.append(nrse_loop)
nrse = pd.DataFrame(nrse_all, columns = np.arange(5,17,1))
print(nrse.agg(['mean']))

selector = RFE(LinReg, 7, 1)
cols = X_train.iloc[:, selector.fit(X_train, Y_train).support_].columns
print (cols)

#Sceond fitting
LinReg2 = LinearRegression()
LinReg2.fit(X_train[cols], Y_train)

LinReg2Pred = LinReg2.predict(X_train[cols])

LinReg2Score = sqrt(mean_squared_error(Y_train, LinReg2Pred))

print('Linear Regression score: ',LinReg2Score)


#Final prediction
FinalPrediction = LinReg2.predict(X_test[cols])

FinalScore = sqrt(mean_squared_error(Y_test, FinalPrediction))

print('Final score: ',FinalScore)
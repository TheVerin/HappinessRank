#WhatMakeUsHappy?

#import libraries
import pandas as pd
import seaborn as sns
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
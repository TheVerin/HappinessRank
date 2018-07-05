#import libraries
import warnings
warnings.filterwarnings('ignore')

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
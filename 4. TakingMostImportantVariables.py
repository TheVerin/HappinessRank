#import libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
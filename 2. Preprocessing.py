#import libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
#import libraries
import warnings
warnings.filterwarnings('ignore')

#Final prediction
FinalPrediction = LinReg2.predict(X_test[cols])

FinalScore = sqrt(mean_squared_error(Y_test, FinalPrediction))

print('Final score: ',FinalScore)
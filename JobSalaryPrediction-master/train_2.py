import data_io
import pickle
import test_feature
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.base import BaseEstimator
from HTMLParser import HTMLParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

##################################################################### trainning
train = data_io.get_train_df()
train_feature=test_feature.get_feature(train)
pca_feature=test_feature.get_pca_feature(train)
print 'training data shape is ', train_feature.shape, ' and ', pca_feature.shape
num_test=10000
training_1=train_feature[:-num_test,]
testing_1=train_feature[-num_test:,]
training_2=pca_feature[:-num_test,]
testing_2=pca_feature[-num_test:,]

print("Extracting features and training model")
classifier_1 = RandomForestRegressor(n_estimators=200, 
                                                verbose=2,
                                                n_jobs=1,
                                            random_state=None,oob_score=True)
classifier_2 = RandomForestRegressor(n_estimators=200, verbose=2, n_jobs=1,random_state=None,oob_score=True)

classifier_1.fit(training_1, train["SalaryNormalized"][:-num_test])
classifier_2.fit(training_2, train["SalaryNormalized"][:-num_test])
print 'fitting socre is ', classifier_1.score(testing_1, train["SalaryNormalized"][-num_test:]), ' and ', classifier_2.score(testing_2, train["SalaryNormalized"][-num_test:])

predictions_1 = classifier_1.predict(testing_1)
predictions_1 = predictions_1.flatten()
orig=train["SalaryNormalized"][-num_test:]
diff_1=np.absolute(predictions_1-orig.flatten())
MAE_1=diff_1.mean()

predictions_2 = classifier_2.predict(testing_2)
predictions_2 = predictions_2.flatten()
diff_2=np.absolute(predictions_2-orig.flatten())
MAE_2=diff_2.mean()

print 'mean absolute avg for selecting feature and reducing feature is ', MAE_1, ' and ', MAE_2


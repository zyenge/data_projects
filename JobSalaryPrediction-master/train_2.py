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
print 'training data shape is ', train_feature.shape
training=train_feature[:-10000,]
testing=train_feature[-10000:,]

print("Extracting features and training model")
classifier = RandomForestRegressor(n_estimators=100, 
                                                verbose=2,
                                                n_jobs=1,
                                            random_state=None,oob_score=True)
classifier.fit(training, train["SalaryNormalized"][:-10000])
print classifier.score(testing, train["SalaryNormalized"][-10000:])

predictions = classifier.predict(testing)
predictions = predictions.flatten()
orig=train["SalaryNormalized"][-10000:]
diff=np.absolute(predictions-orig.flatten())
MAE=diff.mean()
print MAE


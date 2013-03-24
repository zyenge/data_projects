import test_data_io
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
print 'read in raw data'
train = test_data_io.get_train_df()
print 'transform raw data'
train_feature=test_feature.get_feature(train)
pca_feature=test_feature.get_pca_feature(train)
kbest_feature=test_feature.get_kBest_feature(train)
print 'training data shape is ', train_feature.shape, ' and ', pca_feature.shape,' and ', kbest_feature.shape
num_test=1000
training_1=train_feature[:-num_test,]
testing_1=train_feature[-num_test:,]
training_2=pca_feature[:-num_test,]
testing_2=pca_feature[-num_test:,]
fs_box=[(train_feature,'orig_rf'), (pca_feature,'pca_feature'), (kbest_feature,'kbest_feature')]
classifier = RandomForestRegressor(n_estimators=50, 
                                                verbose=2,
                                                n_jobs=2,
                                            random_state=None,oob_score=True)
orig=train["SalaryNormalized"][-num_test:]
score_list=[]
MAE_list=[]
for fs,name in fs_box:
  training=fs[:-num_test,]
  testing=fs[-num_test:,]
  classifier.fit(training, train["SalaryNormalized"][:-num_test])
  score=classifier.score(testing, train["SalaryNormalized"][-num_test:])
  print 'fitting score for', name, ' is ', score
  prediction=classifier.predict(testing)
  prediction=prediction.flatten()
  diff=np.absolute(prediction-orig.flatten())
  MAE=diff.mean()
  print 'mean abs avg for ', name, ' is ', MAE
  MAE_list.append(MAE)
  score_list.append(score)

print fs_box, score_list, MAE_list
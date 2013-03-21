import data_io
import pickle
import test_feature
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

print 'reading training data'
train = data_io.get_train_df()

##################################################################### predicting
print 'reading valid data'
valid = data_io.get_valid_df()
combined=pd.concat([train,valid],ignore_index=True)
print 'feature selecting'
combined_feature=test_feature.get_feature(combined)
train_len=train.shape[0]
x_train=combined_feature[:train_len,]
x_valid=combined_feature[train_len:]
y_train=train["SalaryNormalized"]
classifier = RandomForestRegressor(n_estimators=100, 
                                                verbose=2,
                                                n_jobs=1,
                                            random_state=None,oob_score=True)
print'training model'
classifier.fit(x_train,y_train)

print 'predicting'
predictions_valid = classifier.predict(x_valid)   
predictions_valid = predictions_valid.reshape(len(predictions), 1)

print("Writing predictions to file")
data_io.write_submission(predictions_valid)

print("Saving the classifier")
data_io.save_model(classifier)


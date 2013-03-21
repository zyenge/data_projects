import test_data_io
import pickle
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.base import BaseEstimator
from HTMLParser import HTMLParser
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import cross_validation




# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)

enc = OneHotEncoder()
le = preprocessing.LabelEncoder()
le1 = preprocessing.LabelEncoder()
extracted = []

train = test_data_io.get_train_df()

features = [('FullDescription-Bag of Words', 'FullDescription', CountVectorizer(max_features=100,min_df=1, stop_words='english')),
            ('Title-Bag of Words', 'Title', CountVectorizer(max_features=100,binary=True)),
            ('LocationRaw-Bag of Words', 'LocationRaw', CountVectorizer(max_features=100, min_df=1, stop_words=['uk'],binary=True))]
sub_feature= np.array([le.fit_transform(train['Company']),le1.fit_transform(train['SourceName'])]).T          

enc_sub=enc.fit_transform(sub_feature, y=None).toarray()

for badwords, column, extractor in features:
  extractor.fit(train[column], y=None)
  fea = extractor.transform(train[column])
  if hasattr(fea, "toarray"):
    extracted.append(fea.toarray())
  else:
    extracted.append(fea)

if len(extracted) > 1:
  extracted=np.concatenate(extracted, axis=1)
else: 
  extracted[0]
if len(extracted)==len(enc_sub):
  train_test=np.concatenate((extracted,enc_sub),axis=1)
print train_test.shape, '\n', 'features are ready'


param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
kfold = cross_validation.KFold(len(train_test), n_folds=len(train_test)*0.2)
svc=SVR()
clf = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=-1,cv=kfold,)
clf.fit(train_test, train["SalaryNormalized"]) 
  
print clf.best_score_, clf.best_estimator_, clf.best_params_

# print("Extracting features and training model")
# classifier = SVR(kernel='rbf', C=1e3, gamma=0.1,cv=100)
# classifier.fit(train_test, train["SalaryNormalized"])
# print classifier.score(train_test, train["SalaryNormalized"])

#print("Saving the classifier")
#test_data_io.save_model(classifier)
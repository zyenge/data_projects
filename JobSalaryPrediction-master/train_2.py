import test_data_io
import pickle
import test_feature
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from weight_boosting import AdaBoostRegressor
##################################################################### trainning
num_test=1000
print 'reading training data'
train = test_data_io.get_train_df()

test=train["SalaryNormalized"][:-num_test]
print train.shape[0], test.shape
print 'select features'
train_feature=test_feature.get_feature(train)
#pca_feature=test_feature.get_pca_feature(train)
#np.save('trainingFeatures100k.npy',pca_feature)
#print 'training data shape is ', train_feature.shape, ' and ', pca_feature.shape
pca_feature=np.load('trainingFeatures100k.npy')
pca_feature=pca_feature[:train.shape[0],]
training_1=train_feature[:-num_test,]
testing_1=train_feature[-num_test:,]
training_2=pca_feature[:-num_test,]
testing_2=pca_feature[-num_test:,]

print("Extracting features and training model")
#Tree=DecisionTreeRegressor(max_depth=None)
boostTree=AdaBoostRegressor(DecisionTreeRegressor(max_depth=None), n_estimators=50, random_state=None)
boostSVR=AdaBoostRegressor(SVR(), n_estimators=50, random_state=None)
rf = RandomForestRegressor(n_estimators=50,verbose=2,n_jobs=2,random_state=None,oob_score=True)

MAE_list=[]
score_list=[]
orig=train["SalaryNormalized"][-num_test:]
model_list=['RF','RF_PCA','BoostTree_pca', 'BoostSVR_pca']

#for RF:
print 'fitting rf without pca'
rf.fit(training_1, train["SalaryNormalized"][:-num_test])
score=rf.score(testing_1,train["SalaryNormalized"][-num_test:])
predictions = rf.predict(testing_1)
predictions = predictions.flatten()
diff=np.absolute(predictions-orig.flatten())
MAE=diff.mean()
MAE_list.append(MAE)
score_list.append(score)

classifiers=[rf, boostTree,boostSVR]
for clf in classifiers:
  clf.fit(training_2, train["SalaryNormalized"][:-num_test])
  score=clf.score(testing_2,train["SalaryNormalized"][-num_test:])
  print 'fitting score is ', score
  predictions = clf.predict(testing_2)
  predictions = predictions.flatten()
  diff=np.absolute(predictions-orig.flatten())
  MAE=diff.mean()
  print 'mean abs error is ', MAE
  MAE_list.append(MAE)
  score_list.append(score)
print model_list, score_list, MAE_list

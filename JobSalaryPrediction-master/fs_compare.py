import test_data_io
import pickle
import test_feature
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

##################################################################### trainning
print 'read in raw data'
train = test_data_io.get_train_df()
print 'transform raw data'
train_feature=test_feature.get_feature(train)
pca_feature=test_feature.get_pca_feature(train)
chi2_feature=test_feature.get_chi2_feature(train)
f_feature=test_feature.get_f_feature(train)
kbest_feature=test_feature.get_kBest_feature(train)
#print 'training data shape is ', train_feature.shape, ' and ', pca_feature.shape,' and ', kbest_feature.shape
num_test=5000
fs_box=[(train_feature[0],'top_freq'),(pca_feature[0],'pca'),(kbest_feature[0],'kbest_feature'), (chi2_feature[0],'chi2'), (f_feature[0],'anova f feature')]
classifier = RandomForestRegressor(n_estimators=10, 
                                                verbose=2,
                                                n_jobs=2,
                                            random_state=None,oob_score=True)
orig=train["SalaryNormalized"][-num_test:]
score_list=[]
MAE_list=[]
time_list=[train_feature[1],pca_feature[1],kbest_feature[1],chi2_feature[1],f_feature[1]]
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

comparing_table=np.array([score_list, MAE_list,time_list]).T
np.save('feature_models.npy', comparing_table)
print fs_box, score_list, MAE_list,time_list, 
print 'table size is ', comparing_table.shape
print comparing_table 

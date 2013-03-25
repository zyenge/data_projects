import test_data_io
import pickle
import test_feature
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
##################################################################### trainning
print 'read in raw data'
train = test_data_io.get_train_df()
print 'transform raw data'
#preparing rawdata
le = preprocessing.LabelEncoder()
length=len(train['Company'])
sub_feature=np.concatenate([le.fit_transform(train['Company']),le.fit_transform(train['SourceName'])]).reshape(length,2)
enc_sub=[]
for i, j in sub_feature:
  enc_sub.append([bin(i),bin(j)])
enc_sub=np.array(enc_sub)

num_compo=30
features_0 = [('FullDescription-Bag of Words', train['FullDescription'], CountVectorizer(max_features=num_compo,min_df=1, stop_words='english')),
              ('Title-Bag of Words', train['Title'], CountVectorizer(max_features=num_compo,binary=True,min_df=1)),
              ('LocationRaw-Bag of Words', train['LocationRaw'], CountVectorizer(max_features=num_compo, min_df=1, stop_words=['uk'],binary=True)),
              ('LocationNormalized-Bag of Words', train['LocationNormalized'], CountVectorizer(max_features=num_compo, min_df=1)),
              ('Company words',enc_sub[:,0],CountVectorizer(max_features=num_compo,min_df=1)),
               ('SourceName words',enc_sub[:,1],CountVectorizer(max_features=num_compo,min_df=1))]

features = [('FullDescription-Bag of Words', train['FullDescription'], CountVectorizer(min_df=1, stop_words='english')),
            ('Title-Bag of Words', train['Title'], CountVectorizer(binary=True,min_df=1)),
            ('LocationRaw-Bag of Words', train['LocationRaw'], CountVectorizer(min_df=1, stop_words=['uk'],binary=True)),
            ('LocationNormalized-Bag of Words', train['LocationNormalized'], CountVectorizer(min_df=1)),
            ('Company words',enc_sub[:,0],CountVectorizer(min_df=1)),
             ('SourceName words',enc_sub[:,1],CountVectorizer(min_df=1))]
             
             
train_feature=test_feature.get_feature(train,features_0)
print 'saving features'
np.save('train_feature.npy', train_feature)
pca_feature=test_feature.get_pca_feature(train,features)
print 'saving features'
np.save('pca_feature.npy', pca_feature)
chi2_feature=test_feature.get_chi2_feature(train,features)
print 'saving features'
np.save('chi2_feature.npy', chi2_feature)
f_feature=test_feature.get_f_feature(train,features)
print 'saving features'
np.save('f_feature.npy', f_feature)
kbest_feature=test_feature.get_kBest_feature(train,features)
print 'saving features'
np.save('kbest_feature.npy', kbest_feature)
#print 'training data shape is ', train_feature.shape, ' and ', pca_feature.shape,' and ', kbest_feature.shape
num_test=5000
fs_box=[(train_feature[0],'top_freq'),(pca_feature[0],'pca'),(kbest_feature[0],'kbest_feature'), (chi2_feature[0],'chi2'), (f_feature[0],'anova f feature')]
classifier = RandomForestRegressor(n_estimators=50, 
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

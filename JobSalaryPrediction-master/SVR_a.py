import test_data_io
import pickle
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import numpy as np
from sklearn.base import BaseEstimator
from HTMLParser import HTMLParser
from sklearn.svm import SVR
from sklearn import feature_selection
import numpy.ma as ma

le = preprocessing.LabelEncoder()
extracted = []

print 'reading trainning data'
train = test_data_io.get_train_df()
#sub_feature= np.array([le.fit_transform(train['Company']),le1.fit_transform(train['SourceName'])]).T          
length=len(train['Company'])
sub_feature=np.concatenate([le.fit_transform(train['Company']),le.fit_transform(train['SourceName'])]).reshape(length,2)
enc_sub=[]
for i, j in sub_feature:
  enc_sub.append([bin(i),bin(j)])
enc_sub=np.array(enc_sub)
features = [('FullDescription-Bag of Words', train['FullDescription'], CountVectorizer(max_features=200,min_df=1, stop_words='english')),
            ('Title-Bag of Words', train['Title'], CountVectorizer(max_features=200,binary=True)),
            ('LocationRaw-Bag of Words', train['LocationRaw'], CountVectorizer(max_features=200, min_df=1, stop_words=['uk'],binary=True)),
            ('LocationNormalized-Bag of Words', train['LocationNormalized'], CountVectorizer(max_features=200, min_df=1)),
            ('Company words',enc_sub[:,0],CountVectorizer(max_features=200)),
             ('SourceNane words',enc_sub[:,1],CountVectorizer(max_features=200))]

#enc_sub=enc.fit_transform(sub_feature, y=None).toarray()
#testv=['english engineer soft data','today is a good day','data soft science is hard and easy and I want to be good at it*****','niko is a software engineer who has a big head']
# extractor=CountVectorizer(min_df=1,binary=True)
# extractor.fit(testv,y=None)
# print extractor.get_feature_names()


print 'transform training data'
for badwords, column, extractor in features:
  extractor.fit(column, y=None)
  #print extractor.get_feature_names(), len(extractor.get_feature_names())
  fea = extractor.transform(column)
  if hasattr(fea, "toarray"):
    extracted.append(fea.toarray())
  else:
    extracted.append(fea)

if len(extracted) > 1:
  extracted=np.concatenate(extracted, axis=1)
else: 
  extracted[0]
#if len(extracted)==len(enc_sub):
#  train_test=np.concatenate((extracted,enc_sub),axis=1)

#print 'saving the transformed training data'
#pickle.dump(train_test, open('/Users/zyenge/Documents/python_project/Kaggle/JobSalaryPrediction/train_test.csv', 'w'))
#pickle.dump(train_test, open('~/python/Kaggle/JobSalaryPrediction-master/train_test.csv', 'w'))
test_num=1000
train_test=extracted
training=train_test[:-test_num,]
testing=train_test[-test_num:,]
n_fea=training.shape[1]
print testing.shape
print("Extracting features and training model")
svc=SVR(kernel='rbf', probability=True)
MAE_list=[]

selector=feature_selection.SelectKBest(score_func=feature_selection.f_regression,k=200)
selector.fit(training, train["SalaryNormalized"][:-test_num])
selected_features=selector.transform(training)
index_bool=selector.get_support()
index_num=[]
for i,j in enumerate(index_bool):
  if j:
    index_num.append(i)
start=np.reshape(range(test_num), (test_num,1))
masked_test=start
for i in index_num:
  masked_test=np.concatenate((masked_test,testing[:,i:i+1]),axis=1)
#print 'start is ', masked_test[:,0:1]
masked_test=masked_test[:,1:]
print 'new masked_test shape is ', masked_test.shape

print 'fitting traning'
svc.fit(selected_features,train["SalaryNormalized"][:-test_num])
print svc.score(masked_test, train["SalaryNormalized"][-test_num:])

print 'predicting test data'
predictions = svc.predict(masked_test)
predictions = predictions.flatten()
orig=train["SalaryNormalized"][-test_num:]
diff=np.absolute(predictions-orig.flatten())
MAE=diff.mean()
print MAE

#########################



#svc.fit(training, train["SalaryNormalized"][:-1000])
#fea_no=(n_fea*0.1,n_fea*0.2,n_fea*0.3,n_fea*0.4,n_fea*0.5,n_fea*0.6,n_fea*0.7,n_fea*0.8,n_fea*0.9)

        # transform = feature_selection.SelectPercentile(score_func=feature_selection.RFE)
        # clf = Pipeline([('anova', transform), ('svc', SVR())])
        # 
        # score_means = list()
        # score_stds  = list()
        # percentiles = (10, 15, 20, 30, 40, 60, 80)

########### 
# for fea in fea_no:
#   #clf.set_params(anova__percentile=percentile)
#   selector = feature_selection.f_regression(svc, fea, step=1)
#   this_scores = cross_validation.cross_val_score(selector, training, train["SalaryNormalized"][:-test_num], n_jobs=1)
#   # score_means.append(this_scores.mean())
# #   score_stds.append(this_scores.std())
#   print this_scores
##################
#import pylab as pl 
#pl.errorbar(percentiles, score_means, np.array(score_stds))
# 
# pl.title(
#     'Performance of the SVM-Anova varying the percentile of features selected')
# pl.xlabel('Percentile')
# pl.ylabel('Prediction rate')
# 
# pl.axis('tight')
# pl.show()

# 
# 
# 
# predictions = svc.predict(testing)
# predictions = predictions.flatten()
# orig=train["SalaryNormalized"][-1000:]
# diff=np.absolute(predictions-orig.flatten())
# MAE=diff.mean()
# print MAE

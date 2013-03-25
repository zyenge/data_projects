import data_io
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_selection
from weight_boosting import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.sparse import coo_matrix

print 'reading training data'
train = data_io.get_train_df()
train_len=train.shape[0]
##################################################################### predicting
print 'reading valid data'
valid = data_io.get_valid_df()
combined=pd.concat([train,valid],ignore_index=True)
le = preprocessing.LabelEncoder()
print 'transforming company and sourcename'
length=len(combined['Company'])
sub_feature=np.concatenate([le.fit_transform(combined['Company']),le.fit_transform(combined['SourceName'])]).reshape(length,2)
enc_sub=[]
for i, j in sub_feature:
  enc_sub.append([bin(i),bin(j)])
enc_sub=np.array(enc_sub)

num_compo=200
features = [('FullDescription-Bag of Words', combined['FullDescription'], CountVectorizer(min_df=1, stop_words='english')),
            ('Title-Bag of Words', combined['Title'], CountVectorizer(binary=True,min_df=1)),
            ('LocationRaw-Bag of Words', combined['LocationRaw'], CountVectorizer(min_df=1, stop_words=['uk'],binary=True)),
            ('LocationNormalized-Bag of Words', combined['LocationNormalized'], CountVectorizer(min_df=1)),
            ('Company words',enc_sub[:,0],CountVectorizer(min_df=1)),
             ('SourceName words',enc_sub[:,1],CountVectorizer(min_df=1))]

# get_kBest feature
print 'feature selecting'
selector=feature_selection.SelectKBest(score_func=feature_selection.f_regression,k=num_compo)
selected = np.zeros((length,1))
for bagwords, column, extractor in features:
  extractor.fit(column, y=None)
  fea_kBest = extractor.transform(column)
  print fea_kBest.shape, type(fea_kBest)
  if fea_kBest.shape[1] > num_compo:
    all_list=[]
    kbest_array=np.zeros((length,1))
    for i in range(fea_kBest.shape[1]):
      tup=feature_selection.f_regression(fea_kBest.getcol(i).todense()[:train_len],train["SalaryNormalized"])
      all_list.append(np.array([[tup[1][0],i]]))
    all_array=np.concatenate(all_list)
    sorted_array=all_array[np.lexsort((all_array[:,1],all_array[:,0]))]
    kBest_index=sorted_array[:num_compo,-1:].flatten()
    for i in kBest_index:
      kbest_array=np.concatenate([kbest_array,fea_kBest.getcol(i).todense()],axis=1)
    # selector.fit(fea_kBest[:train_len,:], train["SalaryNormalized"])
#     selected_features=selector.transform(fea_kBest)
    selected=np.concatenate([selected,kbest_array[:,1:]], axis=1)
  else:
    selected=np.concatenate([selected,fea_kBest.toarray()], axis=1)
  print selected.shape

combined_feature=selected[:,1:]
print 'saving kbest features'
x_train=combined_feature[:train_len,]
np.save('Xtrain.npy', x_train)
x_valid=combined_feature[train_len:]
np.save('Xvalid.npy', x_valid)
y_train=train["SalaryNormalized"]
np.save('Ytrain.npy', y_train)

print 'traning shape is ', x_train.shape
print 'valid shape is ', x_valid.shape

classifier=AdaBoostRegressor(DecisionTreeRegressor(max_depth=None), n_estimators=100, random_state=None)
print'training model'
classifier.fit(x_train,y_train)

print 'predicting'
predictions_valid = classifier.predict(x_valid)   
predictions_valid = predictions_valid.reshape(len(predictions_valid), 1)

print("Writing predictions to file")
data_io.write_submission(predictions_valid)

print("Saving the classifier")
data_io.save_model(classifier)


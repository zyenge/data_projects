import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
from sklearn import feature_selection
import time


num_compo=50
           
def get_feature(rawdata,features):
  print 'print feature selecting'
  start=time.clock()
  extracted = []

  for bagwords, column, extractor in features:
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
  eclipsed=time.clock()-start
  return (extracted, eclipsed)

def get_pca_feature(rawdata,features):
  print 'print feature selecting'
  start=time.clock()
  rpca=RandomizedPCA(n_components=num_compo)
  reduced=[]
  for badwords, column, extractor in features:
    extractor.fit(column, y=None)
    fea_ex = extractor.transform(column)
    if fea_ex.shape[1] > num_compo:
      fea_pca=rpca.fit_transform(fea_ex)
      reduced.append(fea_pca)
    else:
      reduced.append(fea_ex.toarray())
  reduced=np.concatenate(reduced, axis=1)
  eclipsed=time.clock()-start
  return (reduced,eclipsed)

def get_kBest_feature(rawdata,features):
  print 'print feature selecting'
  start=time.clock()
  selector=feature_selection.SelectKBest(score_func=feature_selection.f_regression,k=num_compo)
  train_len=rawdata.shape[0]
  selected = np.zeros((train_len,1))
  for badwords, column, extractor in features:
    extractor.fit(column, y=None)
    fea_kBest = extractor.transform(column)
    if fea_kBest.shape[1] > num_compo:
      selector.fit(fea_kBest.toarray(), rawdata["SalaryNormalized"])
      selected_features=selector.transform(fea_kBest.toarray())
      selected=np.concatenate([selected,selected_features], axis=1)
  eclipsed=time.clock()-start
  return (selected[:,1:],eclipsed)

def get_chi2_feature(rawdata,features):
  print 'print feature selecting'
  start=time.clock()
  selector=feature_selection.SelectKBest(score_func=feature_selection.chi2,k=num_compo)
  train_len=rawdata.shape[0]
  selected = np.zeros((train_len,1))
  for badwords, column, extractor in features:
    extractor.fit(column, y=None)
    fea_kBest = extractor.transform(column)
    if fea_kBest.shape[1] > num_compo:
      selector.fit(fea_kBest.toarray(), rawdata["SalaryNormalized"])
      selected_features=selector.transform(fea_kBest.toarray())
      selected=np.concatenate([selected,selected_features], axis=1)
  eclipsed=time.clock()-start
  return (selected[:,1:], eclipsed)

def get_f_feature(rawdata,features):
  print 'print feature selecting'
  start=time.clock()
  selector=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=num_compo)
  train_len=rawdata.shape[0]
  selected = np.zeros((train_len,1))
  for badwords, column, extractor in features:
    extractor.fit(column, y=None)
    fea_kBest = extractor.transform(column)
    if fea_kBest.shape[1] > num_compo:
      selector.fit(fea_kBest.toarray(), rawdata["SalaryNormalized"])
      selected_features=selector.transform(fea_kBest.toarray())
      selected=np.concatenate([selected,selected_features], axis=1)
  eclipsed=time.clock()-start
  return (selected[:,1:], eclipsed)

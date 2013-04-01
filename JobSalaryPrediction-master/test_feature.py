import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
from sklearn import feature_selection
import time
from scipy.sparse import *
import sys


num_compo=100
           
def get_feature(rawdata,features):
  print ' feature selecting'
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
  print ' feature selecting'
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

def get_kBest_feature(rawdata,features,num_compo):
  print ' feature selecting'
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
  print ' feature selecting'
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
  print ' feature selecting'
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

def newk_best(rawdata, features):
  print 'new k best feature selecting'
  start=time.clock()
  selected = np.zeros((num,1))
  for bagwords, column, extractor in features:
    extractor.fit(column, y=None)
    fea_kBest = extractor.transform(column)
    fea_kBest=fea_kBest.tocsc()
    #print 'all feature shape is ', fea_kBest.shape
    p_valueList=[num, bagwords]
    if fea_kBest.shape[1] > num_compo:
      all_list=[]
      kbest_array=np.zeros((num,1))
      for i in range(fea_kBest.shape[1]):
        tup=feature_selection.f_regression(fea_kBest.getcol(i).todense()[:-num_test],rawdata["SalaryNormalized"][:-num_test])
        if tup[1][0] <=1 and tup[1][0]>0.0001:
          all_list.append(np.array([[tup[1][0],i]]))
      all_array=np.concatenate(all_list)
      #print all_array.shape
      sorted_array=all_array[np.lexsort((all_array[:,1],all_array[:,0]))]
      p=2
      kBest_index=[]
      for i,j in enumerate(sorted_array[:,0]):
        v=sorted_array[-i-1,0]
        if abs(v-p)<0.0001:
          continue
        else:
          p_valueList.append(v)
          p=v
          kBest_index.append(sorted_array[-i-1,1])
        if len(kBest_index)==num_compo:
          break
      print 'number of features selected is ', len(kBest_index)
      print 'the p values are ', p_valueList
      writeout="/home/zhen/python/Kaggle/pValue/p_value_%i_%s.npy" %(num, bagwords)
      np.save(writeout, p_valueList)
      for i in kBest_index:
        kbest_array=np.concatenate([kbest_array,fea_kBest.getcol(i).todense()],axis=1)
      selected=np.concatenate([selected,kbest_array[:,1:]], axis=1)
    else:
      selected=np.concatenate([selected,fea_kBest.toarray()], axis=1)
    print selected.shape
  kbest_feature=selected[:,1:]
  eclipse=time.clock()-start
  return (kbest_feature, eclipse)
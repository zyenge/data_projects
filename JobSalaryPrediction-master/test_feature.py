import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import RandomizedPCA



def get_feature(rawdata):
  
  le = preprocessing.LabelEncoder()
  length=len(rawdata['Company'])
  sub_feature=np.concatenate([le.fit_transform(rawdata['Company']),le.fit_transform(rawdata['SourceName'])]).reshape(length,2)
  enc_sub=[]
  for i, j in sub_feature:
    enc_sub.append([bin(i),bin(j)])
  enc_sub=np.array(enc_sub)
  extracted = []
  features = [('FullDescription-Bag of Words', rawdata['FullDescription'], CountVectorizer(max_features=200,min_df=1, stop_words='english')),
              ('Title-Bag of Words', rawdata['Title'], CountVectorizer(max_features=200,binary=True)),
              ('LocationRaw-Bag of Words', rawdata['LocationRaw'], CountVectorizer(max_features=200, min_df=1, stop_words=['uk'],binary=True)),
              ('LocationNormalized-Bag of Words', rawdata['LocationNormalized'], CountVectorizer(max_features=200, min_df=1)),
              ('Company words',enc_sub[:,0],CountVectorizer(max_features=200)),
               ('SourceName words',enc_sub[:,1],CountVectorizer(max_features=200))]
  
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
  return extracted

def get_pca_feature(rawdata):
  num_compo=200
  rpca=RandomizedPCA(n_components=num_compo)
  reduced=[]
  features = [('FullDescription-Bag of Words', rawdata['FullDescription'], CountVectorizer(min_df=1, stop_words='english')),
              ('Title-Bag of Words', rawdata['Title'], CountVectorizer(binary=True)),
              ('LocationRaw-Bag of Words', rawdata['LocationRaw'], CountVectorizer(min_df=1, stop_words=['uk'],binary=True)),
              ('LocationNormalized-Bag of Words', rawdata['LocationNormalized'], CountVectorizer(min_df=1)),
              ('Company words',rawdata['Company'],CountVectorizer()),
               ('SourceName words',rawdata['SourceName'],CountVectorizer())]
  for badwords, column, extractor in features:
    extractor.fit(column, y=None)
    fea_ex = extractor.transform(column)
    if fea_ex.shape[1] > num_compo:
      fea_pca=rpca.fit_transform(fea_ex)
      reduced.append(fea_pca)
    else:
      reduced.append(fea_ex.toarray())
  reduced=np.concatenate(reduced, axis=1)
  return reduced

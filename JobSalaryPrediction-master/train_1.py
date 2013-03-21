import test_data_io
from features import FeatureMapper, SimpleTransform
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


features = [('FullDescription-Bag of Words', 'FullDescription', CountVectorizer(max_features=100)),
                ('Title-Bag of Words', 'Title', CountVectorizer(max_features=100)),
                ('LocationRaw-Bag of Words', 'LocationRaw', CountVectorizer(max_features=100)),
                ('LocationNormalized-Bag of Words', 'LocationNormalized', CountVectorizer(max_features=100)),
                ('Company-Bag of Words', 'Company', CountVectorizer(max_features=100)),
                ('SourceName-Bag of Words', 'SourceName', CountVectorizer(max_features=100))]
combined = FeatureMapper(features)
   
train=test_data_io.get_train_df()
extract_features=combined.fit_transform(X=train,y=None)
training=extract_features[:-1000,]
testing=extract_features[-1000:,]

print("Extracting features and training model")
classifier =RandomForestRegressor(n_estimators=50, 
                                                    verbose=2,
                                                    n_jobs=1,
                                                    random_state=None)
    
classifier.fit(training, train["SalaryNormalized"][:-1000])
print classifier.score(testing, train["SalaryNormalized"][-1000:])



predictions = classifier.predict(testing)
predictions = predictions.flatten()
orig=train["SalaryNormalized"][-1000:]
diff=np.absolute(predictions-orig.flatten())
MAE=diff.mean()
print MAE



# print("Saving the classifier")
# data_io.save_model(classifier)
#     
# if __name__=="__main__":
#     main()
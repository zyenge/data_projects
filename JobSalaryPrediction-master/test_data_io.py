import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def identity(x):
    return x

# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             , "Company": identity
             , "SourceName": identity
             }

#valid_path = get_paths()["train_data_path"]
#vdf=pd.read_csv(valid_path, converters=converters,nrows=6000)
#writer = open('/Users/zyenge/Documents/python_project/Kaggle/Prediction/givenSalary.csv', 'w')
#vdf['SalaryNormalized'][:-1000].to_csv(writer)

def get_train_df():
    train_path = get_paths()["train_data_path"]
    return pd.read_csv(train_path, converters=converters,nrows=10000)

# def get_valid_df():
#     valid_path = get_paths()["train_data_path"]
#     vdf=pd.read_csv(valid_path, converters=converters,nrows=6000)
#     return vdf[:-1000]

def save_model(model):
    out_path = get_paths()["test_model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["test_model_path"]
    return pickle.load(open(in_path))

def write_submission(predictions):
    prediction_path = get_paths()["test_prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["Id"], predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data_df = pd.read_csv('features.csv', header=0)

pipe = make_pipeline(StandardScaler(), SVC())
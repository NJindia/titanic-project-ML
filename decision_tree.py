import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data_df = pd.read_csv('features.csv', header=0)

pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier())
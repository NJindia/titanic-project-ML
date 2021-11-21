import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data_df = pd.read_csv('features.csv', header=0)
num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
ct = make_column_transformer(
    (StandardScaler(), num_cols))

pipe = Pipeline(ct, DecisionTreeClassifier())
pipe.fit(data_df.drop(columns=))
import pandas as pd
import sklearn.metrics
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data_df = pd.read_csv('features.csv', header=0)
X = data_df.drop(columns=['Survived'])
y = data_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
ct = make_column_transformer((StandardScaler(), num_cols))

pipe = Pipeline([('scalar', ct), ('clf', SVC())])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print(accuracy)
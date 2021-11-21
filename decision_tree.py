import pandas as pd
import sklearn.metrics
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data_df = pd.read_csv('features.csv', header=0)
X = data_df.drop(columns=['Survived'])
y = data_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
ct = make_column_transformer((StandardScaler(), num_cols))


def report(expected, predicted):
    print(confusion_matrix(expected, predicted))
    print(classification_report(expected, predicted))


# Hyperparameter Selection
best_accuracy = -1
best_n = -1
best_pred = -1
best_actual = -1
for n in range(1, 50):
    pipe = Pipeline([('scalar', ct), ('clf', DecisionTreeClassifier())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n = n
        best_pred = y_pred
        best_actual = y_test
    print(n, accuracy)
print(best_n, best_accuracy)
report(best_pred, best_actual)

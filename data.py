import pandas as pd

raw_df = pd.read_csv('data.csv', header=0)
data_df = raw_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data_df['Male'] = data_df.apply(lambda row: 1 if row['Sex'] == 'male' else 0, axis=1)
data_df['Embarked'] = data_df['Embarked'].fillna('S')  # most frequent embark location, the "average"
for location in pd.unique(data_df['Embarked']):
    col_name = 'Embarked_' + str(location)
    data_df[col_name] = data_df.apply(lambda row: 1 if row['Embarked'] == location else 0, axis=1)
data_df = data_df.drop(columns=['Sex', 'Embarked'])
data_df.to_csv('features.csv')

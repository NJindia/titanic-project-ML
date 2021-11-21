import pandas as pd

raw_df = pd.read_csv('data.csv', header=0)
data_df = raw_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data_df['Male'] = data_df.apply(lambda row: 1 if row['Sex'] == 'male' else 0, axis=1)
data_df['Embarked'] = data_df['Embarked'].fillna('S')  # most frequent embark location, the "average"
for location in pd.unique(data_df['Embarked']):
    col_name = 'Embarked_' + str(location)
    data_df[col_name] = data_df.apply(lambda row: 1 if row['Embarked'] == location else 0, axis=1)
data_df = data_df.drop(columns=['Sex', 'Embarked'])
# We will make a guess on age by applying the mean on all empty ages.
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean())
# We're missing one fare value. We can guess this value based on the mean of the fares of other third class people.
third_class = data_df.loc[data_df['Pclass'] == 3]
data_df.at[1043, 'Fare'] = third_class['Fare'].mean()

data_df.to_csv('features.csv')


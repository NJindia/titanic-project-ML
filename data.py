import pandas as pd

data = pd.read_csv('data.csv')
# 263 age values are null. 1014 cabin values are null. 2 embarked values are null.
print(data.isnull().sum())
# We parse ['male','female'] as [0,1].
data['Sex'] = data['Sex'].replace(['male','female'],[0,1])
# The Embarked column represents where the passengers boarded. C is Cherbourg, Q is Queenstown, S is Southampton. Where the passenger boarded shouldn't have an effect on their livelihood.
# It's possible that cabins closer to the lifeboat exit are more likely to survive. However, too many cabin numbers are missing for us to make any meaningful inferences based on cabin number.
# Names don't effect livelihood and cannot be categorized meaningfully in our dataset.
# Similarly, ticket numbers do not effect livelihood. Ticket numbers also do not correspond to cabin numbers, and they can't help us fill in cabin numbers.
data = data.drop(columns=['Embarked','Cabin', 'Ticket','Name'])
# It's likely that age matters ("women and children first").
# We will make a guess on age by applying the mean on all empty ages.
data['Age'] = data['Age'].fillna(data['Age'].mean())
# We're missing one fare value. We can guess this value based on the mean of the fares of other third class people.
third_class = data.loc[data['Pclass'] == 3]
data.at[1043, 'Fare'] = third_class['Fare'].mean()

# Cleaned data!
print(data.isnull().sum())
print(data)
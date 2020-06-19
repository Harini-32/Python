import pandas as pd
train = pd.read_csv('train_preprocessed.csv')
# Finding the correlation (statistical summary of the relationship)
print("Correlation:", train['Survived'].corr(train['Sex']))
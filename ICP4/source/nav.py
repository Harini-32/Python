from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score

# reading the data
train_df = pd.read_csv('glass.csv')
X = train_df.drop("Type",axis=1)
Y = train_df["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

NBModel = GaussianNB()
NBModel.fit(X_test, y_test)

y_pred = NBModel.predict(X_test)

accuracy_score(y_test,y_pred)
print("classification_report ",metrics.classification_report(y_test,y_pred))
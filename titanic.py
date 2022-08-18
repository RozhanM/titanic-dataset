import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
#-----------------working with the datdset-----------
df = pd.read_csv('titanic.csv')
print(df.head())
df.info()
df.isna().sum()
df.describe()
sns.heatmap(data = df.isna() , yticklabels=False , cbar = False)
cols_drop = ['PassengerId','Name','Cabin','Age']
df.drop(cols_drop, axis=1, inplace = True)
#--------------------making data float--------------
print(df.dtypes)

df['Sex'] = df['Sex'].astype('category')
df['Ticket'] = df['Ticket'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

cat_cols = df.select_dtypes(['category']).columns
df[cat_cols] = df[cat_cols].apply(lambda x : x.cat.codes)
print(df.head())

df = df.astype('float')
#---------------------making data ready----------------
x = df.loc[ : , 'Pclass' :]
y = df['Survived']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#-------------------training and testing the model------------
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

Score ={}
model = SGDClassifier()
for score in ["accuracy", "precision", "recall"] :
    Score[score] = np.mean(cross_val_score(model, x_train , y_train , cv = 7 , scoring=score))
print(Score)

model.fit(x_train , y_train)
y_predict = model.predict(x_test)

from sklearn.metrics import f1_score , precision_score , recall_score , accuracy_score
print("f1 score = " , f1_score(y_test , y_predict))
print("precision score = " , precision_score(y_test , y_predict))
print("recall score =" , recall_score(y_test , y_predict))
print("accuracy score = " , accuracy_score(y_test , y_predict))



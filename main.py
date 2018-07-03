import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


df = pd.read_csv("doc/dataset_new.csv", encoding='ISO-8859-1')
df['date']=pd.to_datetime(df['date'])

#df = shuffle(df)
#df[["neutral","average","Best","Age","Potential","Overall","Value","Wage","rank"]]
#X = df[["Potential","Overall","Value","Wage"]]
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(X)
#df[["Potential","Overall","Value","Wage"]]=x_scaled

train = df[df['date']< "2018-01-01"]
test = df[(df['date'] >= "2018-01-01") & ( df['date'] < "2018-07-03")]
X = df[["Relevant","neutral","average","Best","Age","rank"]]

#df[["neutral","CAM","CB","CDM","CF","CM","GK","LB","LM","LW","LWB","RB","RM","RW","RWB","ST","Best","Age","Potential","Overall","Value","Wage"]]

Ytrain = train['result'].values
Xtrain = train[["Relevant","neutral","average","Best","Age","rank"]] #,"Potential","Overall","Value","Wage"
Ytest = test['result'].values
Xtest = test[["Relevant","neutral","average","Best","Age","rank"]]

# Xtrain, Ytrain = make_regression(shuffle=True,n_features=8, n_informative=8,random_state=0)
# model = RandomForestRegressor()
# model.fit(Xtrain, Ytrain)

model = RandomForestClassifier(n_estimators=1000, max_depth=3)  # max_depth=50,max_features=20)#n_estimators= 1000, random_state=100)#max_features=10,max_depth=5)
model.fit(Xtrain, Ytrain)


print(model.feature_importances_)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

df['predictions'] = model.predict(X)
df['predict_proba'] = model.predict_proba(X)[:, 1]

from sklearn.metrics import confusion_matrix

X_test_result = pd.DataFrame(Xtest)
Y_test_result = Ytest
X_test_result['predictions'] = model.predict(Xtest)

conf_mat = confusion_matrix(Y_test_result, X_test_result['predictions'])

print(conf_mat)
df.to_csv("doc/result.csv")


#df['predict_proba'] = model.predict_proba(X)[:, 1]
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

Ytrain = train['result'].values
Xtrain = train[["Relevant","neutral","average","Best","Age","rank"]]
Ytest = test['result'].values
Xtest = test[["Relevant","neutral","average","Best","Age","rank"]]


model = svm.SVC(kernel='linear')
model.fit(Xtrain, Ytrain)


#print(model.feature_importances_)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

df['predictions_svm'] = model.predict(X)
#df['predict_proba'] = model.predict_proba(X)[:, 1]
X_test_result = pd.DataFrame(Xtest)
Y_test_result = Ytest
X_test_result['predictions'] = model.predict(Xtest)

conf_mat = confusion_matrix(Y_test_result, X_test_result['predictions'])
print(conf_mat)

df.to_csv("doc/result.csv")
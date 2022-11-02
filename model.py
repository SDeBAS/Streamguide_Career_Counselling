import pandas as pd
import numpy as np
import pickle


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report

dataset=pd.read_csv("Dataset_Final.csv")

X = np.array(dataset.iloc[:, 0:17]) #X is questions
y = np.array(dataset.iloc[:, 17]) #Y is stream



dataset.replace({'target':{'Science':0,'Commerce':1,'Arts / Humanities':2}},inplace=True)
X = dataset.drop(columns=['Username'],axis=1)
X = X.drop(columns=['age'],axis=1)
X = X.drop(columns=['gender'],axis=1)
X = X.drop(columns=['name'],axis=1)
Y = dataset['target']
X = X.drop(columns=['target'],axis=1)
#X = X.drop(columns=['Unnamed:21'],axis=1)
#X.replace({'target':{'Science':0,'Commerce':1,'Arts / Humanities':2}},inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10)


X_train.fillna(3, inplace=True)
X_test.fillna(3, inplace=True)




model = KNeighborsClassifier(n_neighbors=5)

#Fit the model:
model.fit(X_train.values,y_train.values)
  
#Make predictions on training set:
y_pred = model.predict(X_test.values)
  
#Print accuracy
accuracy = metrics.accuracy_score(y_pred,y_test)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
#print ("Mean Squared Error: %s" % "{0:.3%}".format(np.sqrt(mean_squared_error(y_test, y_pred))))


target_names = ['Science', 'Commerce', 'Arts']
print(classification_report(y_test, y_pred, target_names=target_names))

#print(y_pred)
#print(X_test.target)
#new_input=[[3,5,5,5,3,4,5,4,4,3,2,2,3,5,5,5]]
new_input=[[4,1,1,1,3,3,4,2,1,3,2,1,5,4,2,4]]
new_output=model.predict(new_input)
print(new_output)


pickle.dump(model, open('model.pkl','wb'))
print('test file created')
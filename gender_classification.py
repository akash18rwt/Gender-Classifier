from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np

# data train [height,weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#classifiers
dtree_classifier = DecisionTreeClassifier()
svc_classifier = SVC()
knn_classiifer = KNeighborsClassifier()
nb_classifier = GaussianNB()

#Training the models
dtree_classifier.fit(X,Y)
svc_classifier.fit(X,Y)
knn_classiifer.fit(X,Y)
nb_classifier.fit(X,Y)

#Test data
X_test=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

#Predicting the gender
y_pred_dt = dtree_classifier.predict(X_test)
y_pred_svc = svc_classifier.predict(X_test)
y_pred_knn = knn_classiifer.predict(X_test)
y_pred_nb = nb_classifier.predict(X_test)

#Calculating the acuracy
acc_dt = metrics.accuracy_score(Y_test, y_pred_dt)
acc_svc = metrics.accuracy_score(Y_test, y_pred_svc)
acc_knn = metrics.accuracy_score(Y_test, y_pred_knn)
acc_nb = metrics.accuracy_score(Y_test, y_pred_nb)

#the best classifier among decisiontree, svc, knn, NaiveaBayes
index = np.argmax([acc_dt,acc_svc,acc_knn,acc_nb])
best_accuracy = max([acc_dt,acc_svc,acc_knn,acc_nb])
classifiers = {0 : 'Decision Tree',
               1 : 'Support Vector',
               2 : 'Nearesr Neighbor',
               3 : 'Naive Bayes'}
print('Best Gender Classifier is {} with accuracy {} %'.format(classifiers[index], best_accuracy*100))
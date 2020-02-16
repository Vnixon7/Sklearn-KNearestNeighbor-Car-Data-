import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
#NOTE: when classifying data with nearest neighbor,the testing point on graph needs to be odd in order to avoid equal classification.
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

#uploading data
data = pd.read_csv('car.data',sep=',')
print(data.head())

#LabelEncoder used to label data after changing to int
label_maker = preprocessing.LabelEncoder()

#transforming string data into integer data
buying = label_maker.fit_transform(list(data['buying']))
maint = label_maker.fit_transform(list(data['maint']))
doors = label_maker.fit_transform(list(data['doors']))
persons = label_maker.fit_transform(list(data['persons']))
lug_boot = label_maker.fit_transform(list(data['lug_boot']))
safety = label_maker.fit_transform(list(data['safety']))
clash = label_maker.fit_transform(list(data['class']))

#specifying what we will predict
predict = 'class'

#zipping the training data together
x = list(zip(buying,maint,doors,persons,lug_boot,safety))

#prediction
y = list(clash)

#Train Test Split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Training Model#
'''best = .1000
for i in range(20000):
    #training function with SKLEARN ###Test size is used for the amount of data tested. 0.2 for example will test more,sacrificing performance.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    ###TESTING POINT####
    model = KNeighborsClassifier(n_neighbors=9)
    ###TRAINING MODEL###
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    print('interation: ', i, 'Accuracy: ', acc * 100)

    if acc > best:
        best = acc
        with open('KNN_model.pickle', 'wb') as f:
            pickle.dump(model, f)
            print(best)
        break'''


#Uploading pickel(best training model)
load_in = open('KNN_model.pickle', 'rb')
model = pickle.load(load_in)
predicted = model.predict(x_test)
names = ['unacc','acc','good','vgood']

#Printing comparison between prediction, and actual
for i in range(len(predicted)):
    print('Predicted: ',names[predicted[i]], 'Data: ',x_test[i], 'Actual: ',names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9, True)
    ###print('N:',n)###Visual of Arays


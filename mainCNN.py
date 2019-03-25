import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop
from keras.preprocessing import image

import tensorflow as tf


##Reading Data

def contrast(x):
    if float(x)>205:
        return 1
    elif float(x)<50:
        return 0
    else:
        return float(x)/255

train = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\augmented_data.csv', converters = dict([(i+1, contrast) for i in range(28*28)]))
#train = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\train.csv')
print(train.shape)
train.head()

test= pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\test.csv', converters = dict([(i, contrast) for i in range(28*28)]))
#test= pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\test.csv')
print(test.shape)
test.head()

##Data Preprocessing

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')

y_train= to_categorical(y_train)

##Display data repartition

"""repartition =[0 for i in range(10)]
for i in range(y_train.shape[0]):
    repartition[list(y_train[i,:]).index(max(y_train[i,:]))]+=1
sections=[i for i in range(10)]
plt.bar(sections, repartition, align='center', alpha=0.7)
plt.show()"""

##Reshaping Data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

##
def cnn(X_train, y_train, X_test):
    
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose = 0)
    return model.predict(X_test)
    
def nn(X_train, y_train, X_test):
    model= Sequential()
    model.add(Dense(64, activation='relu', input_dim = 28*28))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.optimizer.lr=0.01
    model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose = 0)
    return model.predict(X_test)

##Testing on a homemade example
"""my_tests = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\tests_perso\\my_tests.csv')
print(my_tests.shape)
my_tests.head()

X_my_tests = (my_tests.iloc[:,:784].values).astype('float32') # all pixel values

print(X_my_tests)

print(nn(X_train, y_train, X_my_tests))"""
##
def c_validation(k):
    kf = cross_validation.KFold(X_train.shape[0], n_folds=k)
    
    totalsuccess = 0

    for trainIndex, testIndex in kf:
        trainSet = X_train[trainIndex]
        testSet = X_train[testIndex]
        trainLabels = y_train[trainIndex]
        testLabels = y_train[testIndex]
        
        predictedLabels = nn(trainSet, trainLabels, testSet)
    
        success = 0
        for i in range(testSet.shape[0]):
            if list(predictedLabels[i,:]).index(max(predictedLabels[i,:])) == list(testLabels[i,:]).index(max(testLabels[i,:])):
                success+=1
        print (100*success/testSet.shape[0], '%')
        totalsuccess += success

        
    print (100*totalsuccess/X_train.shape[0], '%')
    return
    
#c_validation(5)

##
y_augmented = np.vstack((y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train))
##
y_test = cnn(augmented, y_augmented, X_test)
np.savetxt('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\predictionCNNepochs10augmented.csv', np.array([[i+1, int(list(y_test[i,:]).index(max(y_test[i,:])))] for i in range(y_test.shape[0])]),fmt='%i', delimiter = ',',newline='\n', header = 'ImageId,Label', comments='')

 
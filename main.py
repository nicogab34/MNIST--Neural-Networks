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
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from keras.optimizers import RMSprop
from keras.preprocessing import image


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def contrast(x):
    if float(x)>205:
        return 1
    elif float(x)<50:
        return 0
    else:
        return float(x)/255

train = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\train.csv', converters = dict([(i+1, contrast) for i in range(28*28)]))
#train = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\train.csv')
print(train.shape)
train.head()

test= pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\test.csv', converters = dict([(i, contrast) for i in range(28*28)]))
#test= pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\test.csv')
print(test.shape)
test.head()

##

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')
    
y_train= to_categorical(y_train)

##
repartition =[0 for i in range(10)]
for i in range(y_train.shape[0]):
    repartition[list(y_train[i,:]).index(max(y_train[i,:]))]+=1
sections=[i for i in range(10)]
plt.bar(sections, repartition, align='center', alpha=0.7)
plt.show()
    
##
"""X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)"""



##
def nn(X_train, y_train, X_test):
    mean_px = X_train.mean().astype(np.float32)
    std_px = X_train.std().astype(np.float32)
    
    def standardize(x):
        return (x-mean_px)/std_px
    
    seed = 43
    np.random.seed(seed)
    model= Sequential()
    #model.add(Lambda(standardize))
    model.add(Dense(128, activation='relu', input_dim=28*28))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.optimizer.lr=0.01
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose = 0)
    return model.predict(X_test)

##
"""p=model.predict(X_val)
correct=0
for i in range(p.shape[0]):
    if list(p[i,:]).index(max(p[i,:]))== list(y_val[i,:]).index(max(y_val[i,:])):
        correct+=1
print(correct/p.shape[0])"""
##
my_tests = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\tests_perso\\my_tests.csv')
print(my_tests.shape)
my_tests.head()

X_my_tests = (my_tests.iloc[:,:784].values).astype('float32') # all pixel values

print(X_my_tests)

print(nn(X_train, y_train, X_my_tests))
##
def c_validation(k):
    kf = cross_validation.KFold(X_train.shape[0], n_folds=k)
    wrong=[]
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
            else:
                wrong.append((testSet[i], testLabels[i]))
        print (100*success/testSet.shape[0], '%')
        totalsuccess += success

        
    print (100*totalsuccess/X_train.shape[0], '%')
        
    return wrong
    
wrong = c_validation(5)

##
for i in range(min(len(wrong), 20)):
    plt.subplot(4,5,1+i)
    plt.imshow(wrong[i][0].reshape(28,28))
    plt.title(list(wrong[i][1]).index(max(wrong[i][1])))
plt.show()

##
y_test = nn(X_train, y_train, X_test)
np.savetxt('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\prediction128norm2epochs50.csv', np.array([[i+1, int(list(y_test[i,:]).index(max(y_test[i,:])))] for i in range(y_test.shape[0])]),fmt='%i', delimiter = ',',newline='\n', header = 'ImageId,Label', comments='')

 
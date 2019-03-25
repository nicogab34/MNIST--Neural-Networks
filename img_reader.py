from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

test = ndimage.imread('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\tests_perso\\3_2.png',mode = 'L')    
plt.imshow(test,cmap=plt.cm.gray)
plt.show()
print(test.shape, test.dtype)
print(test)

test = test.reshape((1,784))
testaux = np.zeros((1,785))
testaux[:,:784] = test
testaux[0,784] = 4
print(testaux.shape)

pd.DataFrame(testaux,columns = (['pixel'+str(i) for i in range(784)]+['label'])).to_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\tests_perso\\my_tests.csv', index = False)
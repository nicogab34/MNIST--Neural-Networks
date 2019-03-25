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

train = pd.read_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\train.csv', converters = dict([(i+1, contrast) for i in range(28*28)]))
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

##Display data repartition

"""repartition =[0 for i in range(10)]
for i in range(y_train.shape[0]):
    repartition[list(y_train[i,:]).index(max(y_train[i,:]))]+=1
sections=[i for i in range(10)]
plt.bar(sections, repartition, align='center', alpha=0.7)
plt.show()"""

##Data augmentation

def convert_2d(x):
    """x: 2d numpy array. m*n data image.
       return a 3d image data. m * height * width * channel."""
    if len(x.shape) == 1:
        m = 1
        height = width = int(np.sqrt(x.shape[0]))
    else:
        m = x.shape[0]
        height = width = int(np.sqrt(x.shape[1]))

    x_2d = np.reshape(x, (m, height, width, 1))
    
    return x_2d

def crop_image(x, y, min_scale):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       min_scale: float. The minimum scale for cropping.
       return zoomed images.
       This function crops the image, enlarges the cropped part and uses it as augmented data."""
    # convert the data to 2-d image. images should be a m*h*w*c numpy array.
    images = convert_2d(x)
    # m is the number of images. Since this is a gray-scale image scale from 0 to 255, it only has one channel.
    m, height, width, channel = images.shape
    
    # tf tensor for original images
    img_tensor = tf.placeholder(tf.int32, [1, height, width, channel])
    # tf tensor for 4 coordinates for corners of the cropped image
    box_tensor = tf.placeholder(tf.float32, [1, 4])
    box_idx = [0]
    crop_size = np.array([height, width])
    # crop and resize the image tensor
    cropped_img_tensor = tf.image.crop_and_resize(img_tensor, box_tensor, box_idx, crop_size)
    # numpy array for the cropped image
    cropped_img = np.zeros((m, height, width, 1))

    with tf.Session() as sess:

        for i in range(m):
            
            # randomly select a scale between [min_scale, min(min_scale + 0.05, 1)]
            rand_scale = np.random.randint(min_scale * 100, np.minimum(min_scale * 100 + 5, 100)) / 100
            # calculate the 4 coordinates
            x1 = y1 = 0.5 - 0.5 * rand_scale
            x2 = y2 = 0.5 + 0.5 * rand_scale
            # lay down the cropping area
            box = np.reshape(np.array([y1, x1, y2, x2]), (1, 4))
            # save the cropped image
            cropped_img[i:i + 1, :, :, :] = sess.run(cropped_img_tensor, feed_dict={img_tensor: images[i:i + 1], box_tensor: box})
    
    # flat the 2d image
    cropped_img = np.reshape(cropped_img, (m, -1))

    return cropped_img
    
def translate(x, y, dist):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       dist: float. Percentage of height/width to shift.
       return translated images.
       This function shift the image to 4 different directions.
       Crop a part of the image, shift it and fill the left part with 0."""
    # convert the 1d image data to a m*h*w*c array
    images = convert_2d(x)
    m, height, width, channel = images.shape
    
    # set 4 groups of anchors. The first 4 int in a certain group lay down the area we crop.
    # The last 4 sets the area to be moved to. E.g.,
    # new_img[new_top:new_bottom, new_left:new_right] = img[top:bottom, left:right]
    anchors = []
    anchors.append((0, height, int(dist * width), width, 0, height, 0, width - int(dist * width)))
    anchors.append((0, height, 0, width - int(dist * width), 0, height, int(dist * width), width))
    anchors.append((int(dist * height), height, 0, width, 0, height - int(dist * height), 0, width))
    anchors.append((0, height - int(dist * height), 0, width, int(dist * height), height, 0, width))
    
    # new_images: d*m*h*w*c array. The first dimension is the 4 directions.
    new_images = np.zeros((4, m, height, width, channel))
    for i in range(4):
        # shift the image
        top, bottom, left, right, new_top, new_bottom, new_left, new_right = anchors[i]
        new_images[i, :, new_top:new_bottom, new_left:new_right, :] = images[:, top:bottom, left:right, :]
    
    new_images = np.reshape(new_images, (4 * m, -1))
    y = np.tile(y, (4, 1)).reshape((-1, 1))

    return new_images
    
def add_noise(x, y, noise_lvl):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       noise_lvl: float. Percentage of pixels to add noise in.
       return images with white noise.
       This function randomly picks some pixels and replace them with noise."""
    m, n = x.shape
    # calculate the # of pixels to add noise in
    noise_num = int(noise_lvl * n)

    for i in range(m):
        # generate n random numbers, sort it and choose the first noise_num indices
        # which equals to generate random numbers w/o replacement
        noise_idx = np.random.randint(0, n, n).argsort()[:noise_num]
        # replace the chosen pixels with noise from 0 to 255
        x[i, noise_idx] = np.random.randint(0, 255, noise_num)

    noisy_data = x.astype("int")

    return noisy_data
    
def rotate_image(x, y, max_angle):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       max_angle: int. The maximum degree for rotation.
       return rotated images.
       This function rotates the image for some random degrees(0.5 to 1 * max_angle degree)."""
    images = convert_2d(x)
    m, height, width, channel = images.shape
    
    img_tensor = tf.placeholder(tf.float32, [m, height, width, channel])
    
    # half of the images are rotated clockwise. The other half counter-clockwise
    # positive angle: [max/2, max]
    # negative angle: [360-max/2, 360-max]
    rand_angle_pos = np.random.randint(max_angle / 2, max_angle, int(m / 2))
    rand_angle_neg = np.random.randint(-max_angle, -max_angle / 2, m - int(m / 2)) + 360
    rand_angle = np.transpose(np.hstack((rand_angle_pos, rand_angle_neg)))
    np.random.shuffle(rand_angle)
    # convert the degree to radian
    rand_angle = rand_angle / 180 * np.pi
    
    # rotate the images
    rotated_img_tensor = tf.contrib.image.rotate(img_tensor, rand_angle)

    with tf.Session() as sess:
        rotated_imgs = sess.run(rotated_img_tensor, feed_dict={img_tensor: images})
    
    rotated_imgs = np.reshape(rotated_imgs, (m, -1))
    
    return rotated_imgs
    
print("Augment the data...")
cropped_imgs = crop_image(X_train, y_train, 0.9)
translated_imgs = translate(X_train, y_train, 0.1)
noisy_imgs = add_noise(X_train, y_train, 0.1)
rotated_imgs = rotate_image(X_train, y_train, 10)

augmented = np.vstack((X_train, cropped_imgs, translated_imgs, noisy_imgs, rotated_imgs))
#augmented is the new data composed with : the basic dataset, the cropped dataset, the shifted dataset (1 for each 4 directions), the noisy dataset and the rotated dataset

print("Done!", X_train.shape)

##Save

header = 'label'
for i in range(784):
    header+=',pixel'+str(i)

np.savetxt('C:\\Users\\Nicolas\\Desktop\\augmented_data.csv', np.array([[int(y_train[i%X_train.shape[0]])]+list(augmented[i,:]) for i in range(augmented.shape[0])]),fmt='%i', delimiter = ',',newline='\n', header = header, comments='')

##Save2
pd.DataFrame(augmented).to_csv('I:\\Centrale\\Machine Learning\\Kaggle-3-MNIST\\augmented_data.csv', index = False)
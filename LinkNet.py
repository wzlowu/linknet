#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from matplotlib import image
import matplotlib as plt
import matplotlib.pyplot as pltp
import cv2
from PIL import Image
import numpy as np
import os
from tensorflow.keras import layers, optimizers, Sequential
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Conv2D, Activation, UpSampling2D
from labels import name2label
import labels
import time


# In[2]:


# Print entire array
np.set_printoptions(threshold=np.inf)


# In[3]:


def walk_dir(root):
    file_list=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_list.append(os.path.join(path, name))
    return file_list

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, input_size)
    x = x.astype(np.float32)
    x = x /255.0
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, input_size)
    
    x=x.astype(np.int32)
    return x

def encode_label(mask, dictionary):
    label_mask = np.zeros_like(mask)
    for k in dictionary:
        label_mask[mask == k] = dictionary[k]
    return label_mask

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset


# In[4]:


classes = [
    'unlabeled', 
    'road',
    'sidewalk',
    'building',
    'traffic light',
    'traffic sign',
    'vegetation',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'bicycle'
]

id_name = {}
for index, name in enumerate(classes):
    id_name.update({index:name})
for index, name in enumerate(classes):
    id_name.update({name:index})

map_14={}
for i in range(0,33):
    map_14.update({i:0})

map_14.update({-1:0})


for name in classes:
    map_14[name2label[name].id] = id_name[name]


# In[ ]:


#--------Testing Block
input_size = (512,256)
batch_size=25

c=[]
d=[]

leftImg8Bit = './data/leftImg8bit/'
gt_fine = './data/gtFine/'
a = walk_dir(leftImg8Bit + 'val')
e = walk_dir(gt_fine + 'val')

b = [label for label in e if label.endswith('_labelIds.png')]

for imgs in a:
    c.append(read_image(imgs))
for label in b:
    d.append(read_image(label))

testing_data1 = tf_dataset(c, d, batch_size)

print('image list shape: ', np.array(c).shape)
print('mask list shape: ', np.array(d).shape)


# In[ ]:


#---------Testing Block
print('input_data type: ', type(testing_data1))
print('testing list type: ', type(testing))
shape = 0
for x, y in testing_data1:
    data_shape = x.shape
    mask_shape = y.shape
    retr = x
    break
print('image shape: ', data_shape)
print('mask shape: ', mask_shape)
print('image list shape: ', np.array(c).shape)
print('mask list shape: ', np.array(d).shape)
print(np.array(c[0]).shape)
print(type(c[0]))


# In[ ]:


#--------Testing Block
data = c[0]*255
data = data.astype(np.uint8)
data = Image.fromarray(data)
data.show()


# In[ ]:


#--------Testing Block
image = Image.open(a[0])
image.show()
image2 = np.array(image)
image2 = image2.astype(np.float32)
image2 = image2 / 255
image2 = image2 * 255
image2 = image2.astype(np.uint8)
image2 = Image.fromarray(image2)
image2.show()
print(type(image))
print(type(image2))


# In[5]:


#-----------Load Data
# Get execution time for loading and preprocessing data
begin = time.time()

input_size = (256,512)
batch_size=25

train_ds = []
val_ds = []
test_ds = []
color_ds =[]

train_mask=[]
val_mask=[]
test_mask=[]

#Load Data from CityScapes Dataset
leftImg8Bit = './data/leftImg8bit/'
gt_fine = './data/gtFine/'

#Images List
train_imgs = walk_dir(leftImg8Bit + 'train')
val_imgs = walk_dir(leftImg8Bit + 'val')
test_imgs = walk_dir(leftImg8Bit + 'test')

#Label Images List
train_label_list = walk_dir(gt_fine + 'train')
train_label_list = [label for label in train_label_list if label.endswith('_labelIds.png')]

val_label_list = walk_dir(gt_fine + 'val')
val_label_list = [label for label in val_label_list if label.endswith('_labelIds.png')]

test_label_list = walk_dir(gt_fine + 'test')
test_label_list = [label for label in test_label_list if label.endswith('_labelIds.png')]

print("begin preprocessing and loading TRAIN DATA")
begin_data = time.time()
for imgs in train_imgs:
    train_ds.append(read_image(imgs))
for imgs in train_label_list:
    train_mask.append(read_mask(imgs))
    
for i, mask in enumerate(train_mask):
    train_mask[i] = encode_label(mask, map_14)
    
input_data = tf_dataset(train_ds, train_mask, batch_size)
time_data = time.time() - begin_data
print("Done preprocessing and loading TRAIN DATA")
print("Training Data preprocessing and loading takes ", time_data, "seconds" )

train_steps=len(train_ds)//batch_size

del train_ds, train_mask
#--------------------------------

print("begin preprocessing and loading VALIDATION DATA")
begin_data = time.time()
for imgs in val_imgs:
    val_ds.append(read_image(imgs))
for imgs in val_label_list:
    val_mask.append(read_mask(imgs))
    
for i, mask in enumerate(val_mask):
    val_mask[i] = encode_label(mask, map_14)
    
val_data = tf_dataset(val_ds, val_mask, batch_size)
time_data = time.time() - begin_data
print("Done preprocessing and loading VALIDATION DATA")
print("Validation Data preprocessing and loading takes ", time_data, "seconds" )

val_steps = len(val_ds)//batch_size

del val_ds, val_mask
#--------------------------------

print("begin preprocessing and loading TEST DATA")
begin_data = time.time()
for imgs in test_imgs:
    test_ds.append(read_image(imgs))
for imgs in test_label_list:
    test_mask.append(read_mask(imgs))
for i, mask in enumerate(test_mask):
    test_mask[i] = encode_label(mask, map_14)

test_data = tf_dataset(test_ds, test_mask, batch_size)
time_data = time.time() - begin_data
print("Done preprocessing and loading TEST DATA")
print("Test Data preprocessing and loading takes ", time_data, "seconds" )

del test_ds, test_mask, begin_data, time_data

#Colored Images List
train_label_color = walk_dir(gt_fine + 'train')
train_label_color = [label for label in train_label_color if label.endswith('_color.png')]

duration = time.time() - begin
print("Total data preprocessing and Loading takes ", duration, "seconds")
del begin, duration


# In[ ]:


#----------Testing Block
for x, y in input_data:
    print(x.shape, y.shape)
    break


# In[ ]:


#----------Testing Block
print('total {} train images'.format(len(train_imgs)))
print('total {} val images'.format(len(val_imgs)))
print('total {} test images'.format(len(test_imgs)))
print('total {} train labels'.format(len(train_mask)))
print('total {} val labels'.format(len(val_mask)))
print('total {} test labels'.format(len(test_mask)))
print('total {} test color images'.format(len(train_label_color)))


# In[6]:


group = [classes, map_14]
print('class name              CityScape Label          New Label')
print(labels.labels[0].name, '\t\t0\t\t\t', map_14[0])
print(labels.labels[1].name, '\t\t1\t\t\t', map_14[1])
print(labels.labels[2].name, '\t2\t\t\t', map_14[2])
print(labels.labels[3].name, '\t\t3\t\t\t', map_14[3])
print(labels.labels[4].name, '\t\t\t4\t\t\t', map_14[4])
print(labels.labels[5].name, '\t\t5\t\t\t', map_14[5])
print(labels.labels[6].name, '\t\t\t6\t\t\t', map_14[6])
print(labels.labels[7].name, '\t\t\t7\t\t\t', map_14[7])
print(labels.labels[8].name, '\t\t8\t\t\t', map_14[8])
print(labels.labels[9].name, '\t\t9\t\t\t', map_14[9])
print(labels.labels[10].name, '\t\t10\t\t\t', map_14[10])
print(labels.labels[11].name, '\t\t11\t\t\t', map_14[11])
print(labels.labels[12].name, '\t\t\t12\t\t\t', map_14[12])
print(labels.labels[13].name, '\t\t\t13\t\t\t', map_14[13])
print(labels.labels[14].name, '\t\t14\t\t\t', map_14[14])
print(labels.labels[15].name, '\t\t\t15\t\t\t', map_14[15])
print(labels.labels[16].name, '\t\t\t16\t\t\t', map_14[16])
print(labels.labels[17].name, '\t\t\t17\t\t\t', map_14[17])
print(labels.labels[18].name, '\t\t18\t\t\t', map_14[18])
print(labels.labels[19].name, '\t\t19\t\t\t', map_14[19])
print(labels.labels[20].name, '\t\t20\t\t\t', map_14[20])
print(labels.labels[21].name, '\t\t21\t\t\t', map_14[21])
print(labels.labels[22].name, '\t\t22\t\t\t', map_14[22])
print(labels.labels[23].name, '\t\t\t23\t\t\t', map_14[23])
print(labels.labels[24].name, '\t\t\t24\t\t\t', map_14[24])
print(labels.labels[25].name, '\t\t\t25\t\t\t', map_14[25])
print(labels.labels[26].name, '\t\t\t26\t\t\t', map_14[26])
print(labels.labels[27].name, '\t\t\t27\t\t\t', map_14[27])
print(labels.labels[28].name, '\t\t\t28\t\t\t', map_14[28])
print(labels.labels[29].name, '\t\t29\t\t\t', map_14[29])
print(labels.labels[30].name, '\t\t30\t\t\t', map_14[30])
print(labels.labels[31].name, '\t\t31\t\t\t', map_14[31])
print(labels.labels[32].name, '\t\t32\t\t\t', map_14[32])
print(labels.labels[33].name, '\t\t33\t\t\t', map_14[33])
print(labels.labels[-1].name, '\t\t-1\t\t\t', map_14[-1])


# In[7]:


#Each convolution layer follows by a BatchNormalization layer and a RelU non-linearity layer
def Encoder_Block(inputs, filters, pool=False):
    x = Conv2D(filters=filters[0], kernel_size=3, strides=(2,2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=filters[1], kernel_size=3, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=filters[1], kernel_size=3, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=filters[1], kernel_size=3, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


# In[8]:


def Decoder_Block(inputs, filters, pool=False):
    x = Conv2D(filters=filters[0], kernel_size=1, strides=(1,1), padding='same')(inputs)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(filters=filters[0]/4, kernel_size=3, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=filters[1], kernel_size=1, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


# In[9]:


#LinkNet
def linknet(shape, num_class):
    inputs = keras.Input(shape)
    
    #Pre-encode Block
    x0 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same')(inputs)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    x0 = MaxPooling2D(pool_size=(3,3), padding='same', strides=(2,2))(x0)
    
    #Encode
    x1 = Encoder_Block(inputs=x0, filters=(64,64))
    x2 = Encoder_Block(inputs=x1, filters=(64,128))
    x3 = Encoder_Block(inputs=x2, filters=(128,256))
    x4 = Encoder_Block(inputs=x3, filters=(256,512))
    
    #Decode
    x5 = Decoder_Block(inputs=x4, filters=(512,256))
    x6 = Decoder_Block(inputs=(x5+x3), filters=(256,128))
    x7 = Decoder_Block(inputs=(x6+x2), filters=(128,64))
    x8 = Decoder_Block(inputs=(x7+x1), filters=(64,64))
    
    #Post-Decode Block
    x9 = UpSampling2D(size=(2,2))(x8)
    x9 = Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same')(x9)
    x9 = BatchNormalization()(x9)
    x9 = Activation('relu')(x9)
    
    x10 = Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same')(x9)
    x10 = BatchNormalization()(x10)
    x10 = Activation('relu')(x10)
    
    x11 = UpSampling2D(size=(2,2))(x10)
    x11 = Conv2D(filters=num_class, kernel_size=3, strides=(1,1), padding='same')(x11)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)
    
    outputs = x11
    #outputs = layers.Flatten()(x11)
    #outputs = layers.Dense(num_class, activation='softmax')(outputs)
    
    return keras.Model(inputs, outputs)


# In[ ]:


del model


# In[10]:


is_3 = (512, 256, 3)
num_class=len(classes)

epochs=3
lr=0.001

callbacks =[
    keras.callbacks.ModelCheckpoint('Model0_SCCE_METRICS.h5', save_best_only=True, verbose=1)
]

model = linknet(shape=is_3, num_class=num_class)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)



callbacks_1 = [
    keras.callbacks.ModelCheckpoint('Model1_ACCURACY_METRICS.h5', save_best_only=True, verbose=1)
]

model_1 = linknet(shape=is_3, num_class=num_class)
model_1.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
#model.compile(
#    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#    metrics=[tf.keras.metrics.MeanIoU(num_class)]
#)


# In[11]:


begin = time.time()
model.fit(
    input_data,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_data,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1
)

duration = time.time() - begin
print('\n\nModel 0 Training time: ', duration, 'seconds')


# In[ ]:


begin = time.time()
model_1.fit(
    input_data,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks_1,
    validation_data=val_data,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1
)

duration = time.time() - begin
print('\n\nModel 1 Training time: ', duration, 'seconds')


# In[ ]:


#----------Testing Block
model.summary()


# In[ ]:


#----------Testing Block
print(len(model.layers))


# In[ ]:


#----------Testing Block
layer = model.get_layer('activation_1')
print(layer.trainable)


# In[ ]:





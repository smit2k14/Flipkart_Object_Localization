#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("training_set.csv")
data = np.array(data)
print(data)


# In[2]:


from PIL import Image
from PIL import ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[3]:


def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image


# In[4]:


def getDataAndSize():
    id_to_data = []
    indexId = -1
    data = pd.read_csv("training_set.csv")
    data = np.array(data)
    data = data
    for dat in data:
      indexId+=1
      path = dat[0]
      image=Image.open("XceptionNetImages/"+path).convert('RGB')
      image=np.array(image,dtype=np.float32)
      image=image/255
      image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
      id_to_data.append(image)
      print(indexId)
      
    id_to_data=np.array(id_to_data)
    return id_to_data


# In[1]:


def getBoxes():
  id_to_box=[]
  l = []
  indexId = -1
  data = pd.read_csv("training_set.csv")
  data = np.array(data)
  for dat in data:
      indexId+=1
      box = [0,0,0,0]
      box[0]=(dat[1]/2)
      box[1]=(dat[3]/2)
      box[2]=(dat[2]/2)
      box[3]=(dat[4]/2)
      id_to_box.append(box)
      print(indexId)
  del data    
  id_to_box=np.array(id_to_box)
  return id_to_box


# In[6]:


import matplotlib.pyplot as plt
import random

plt.switch_backend('agg')


# In[7]:


def getdata():
    # read data and shuffle
    index=[i for i in range(23990)]
    random.shuffle(index)
    data = getDataAndSize()
    data = np.array(data)
    print(data.shape)
    # print(data)
    data=data[index]
    data_train=data[0:23990]
    data_test=data[23990:]
    box = getBoxes()
    box=box[index]
    box_train=box[0:23990]
    box_test=box[23990:]
    return data_train,box_train,data_test,box_test


# In[8]:


def plot_model(model_details):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_details.history['my_metric'])+1),model_details.history['my_metric'])
    axs[0].plot(range(1,len(model_details.history['val_my_metric'])+1),[1.7*x for x in model_details.history['val_my_metric']])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['my_metric'])+1),len(model_details.history['my_metric'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig("model.png")


# In[9]:


import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten,Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


data_train,box_train,data_test,box_test=getdata()


# In[2]:


def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*224
    x=tf.maximum(tf.minimum(x,224.0),0.0)
    y=predictions[:,1]*224
    y=tf.maximum(tf.minimum(y,224.0),0.0)
    width=predictions[:,2]*224
    width=tf.maximum(tf.minimum(width,320.0),0.0)
    height=predictions[:,3]*224
    height=tf.maximum(tf.minimum(height,224.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.multiply(width,height)
    a2=tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.reduce_mean(sum)


# In[3]:


def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*224)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.where(condition,small_res,large_res)
        loss = tf.reduce_mean(loss)
    return loss


# In[4]:


def resnet_block(inputs,num_filters,kernel_size,strides,activation='relu'):
    x=Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(inputs)
    x=BatchNormalization()(x)
    if(activation):
        x=Activation('relu')(x)
    return x


# In[ ]:


def resnet18():
    inputs=Input((224,224,3))
    
    # conv1
    x=resnet_block(inputs,64,[7,7],2)

    # conv2
    x=MaxPooling2D([3,3],2,'same')(x)
    for i in range(2):
        a=resnet_block(x,64,[3,3],1)
        b=resnet_block(a,64,[3,3],1,activation=None)
        x=keras.layers.add([x,b])
        x=Activation('relu')(x)
    
    # conv3
    a=resnet_block(x,128,[1,1],2)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=Conv2D(128,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,128,[3,3],1)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    # conv4
    a=resnet_block(x,256,[1,1],2)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=Conv2D(256,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,256,[3,3],1)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    # conv5
    a=resnet_block(x,512,[1,1],2)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=Conv2D(512,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,512,[3,3],1)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    x=AveragePooling2D(pool_size=7,data_format="channels_last")(x)
    # out:1*1*512

    y=Flatten()(x)
    # out:512
    y=Dense(1000,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    y=Dropout(0.5,noise_shape=None)(y)
    outputs=Dense(4,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    
    model=Model(inputs=inputs,outputs=outputs)
    return model



# In[ ]:


input_shape=(240,320,3)
model = my_customisable_XceptionNet(input_shape=input_shape,number_of_inner_blocks=1)


# In[ ]:


model.compile(loss = smooth_l1_loss,optimizer=Adam(),metrics=[my_metric])
model.summary()


# In[ ]:


def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5


# In[ ]:


lr_scheduler=LearningRateScheduler(lr_sch)
lr_reducer=ReduceLROnPlateau(monitor='val_my_metric',factor=0.2,patience=5,mode='max',min_lr=1e-3)

checkpoint=ModelCheckpoint('XceptionNet2.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


# In[ ]:


model_details=model.fit(data_train,box_train,batch_size=128,epochs=150,shuffle=True,validation_split=0.1,callbacks=[lr_scheduler,lr_reducer,checkpoint],verbose=1)


# In[ ]:


plot_model(model_details)


# In[ ]:





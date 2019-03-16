#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random
import tensorflow as tf


# In[2]:


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[3]:


plt.switch_backend('agg')


# In[4]:


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


# In[5]:


def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*224
    x=tf.maximum(tf.minimum(x,224.0),0.0)
    y=predictions[:,1]*224
    y=tf.maximum(tf.minimum(y,224.0),0.0)
    width=predictions[:,2]*224
    width=tf.maximum(tf.minimum(width,224.0),0.0)
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


# In[6]:


def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image


# In[7]:


mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]


# In[8]:


def getDataAndSize():
    id_to_data = []
    indexId = -1
    data = pd.read_csv("test_set.csv")
    data = np.array(data)
    for dat in data:
      indexId+=1
      path = dat[0]
      image=Image.open("testResizedImages/"+path).convert('RGB')
      image=image.resize((224,224))
      image=np.array(image,dtype=np.float32)
      image=image/255
      image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
      id_to_data.append(image)
      print(indexId)
      
    id_to_data=np.array(id_to_data)
    return id_to_data


# In[9]:


test_data = getDataAndSize()


# In[10]:


model_resnet50=keras.models.load_model("finalModel.h5",custom_objects={'smooth_l1_loss': smooth_l1_loss,'my_metric':my_metric})
model_resnet18=keras.models.load_model("resnet18model.h5",custom_objects={'smooth_l1_loss': smooth_l1_loss,'my_metric':my_metric})
model_resnet18_2=keras.models.load_model("resnet18model2.h5",custom_objects={'smooth_l1_loss': smooth_l1_loss,'my_metric':my_metric})


# In[11]:


pred = (0.4*model_resnet18.predict(test_data) + 0.4*model_resnet50.predict(test_data)+0.2*model_resnet18_2.predict(test_data))


# In[26]:


pred = pred*224


# In[27]:


width = np.copy(pred[:,2])
height = np.copy(pred[:,3])
x1 = np.copy(pred[:,0])
y1 = np.copy(pred[:,1])


# In[28]:


p = np.copy(pred)


# In[29]:


p[:,0] = x1
p[:,1] = width + x1
p[:,2] = y1
p[:,3] = y1 + height


# In[30]:


p[:,0] = p[:,0]*640/224
p[:,2] = p[:,2]*480/224
p[:,1] = p[:,1]*640/224
p[:,3] = p[:,3]*480/224


# In[31]:


print(p)


# In[32]:


pred_df = pd.DataFrame(p,columns = ['x1','x2','y1','y2'])


# In[33]:


data = pd.read_csv('test_set.csv')


# In[34]:


data = np.array(data)
pred_df = np.array(pred_df)


# In[35]:


for i in range(len(data)):
    data[i,1] = pred_df[i,0]
    data[i,2] = pred_df[i,1]
    data[i,3] = pred_df[i,2]
    data[i,4] = pred_df[i,3]


# In[36]:


final_pred = pd.DataFrame(data, columns = ['image_name','x1','x2','y1','y2'])


# In[37]:


final_pred.to_csv('pred_combined_weight_3.csv', index = False)


# In[ ]:





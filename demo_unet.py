
import pickle

# Getting back the objects:
with open('X.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X = pickle.load(f)
with open('y_label.pkl','rb') as f:  # Python 3: open(..., 'rb')
    y_label = pickle.load(f)


import numpy as np
import cv2
label_size=300

labels=[]
X = (np.array(X,dtype=np.float32) - 128)/128

for i in range(X.shape[0]): 
  label=np.zeros((label_size,label_size,1))
  point=[]
  for i in y_label[0]:
    point.append((int(i[0]),int(i[1])))
  label=cv2.drawContours(label, [np.array(point)], 0, 1, -1)
  labels.append(label)
labels = np.array(labels)

print(X[0])

print(labels[0])

import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import os

import tensorflow as tf
tf.shape(X)

from sklearn.model_selection import train_test_split
X_train,X_validation,y_train,y_validation = train_test_split(X,labels,test_size=0.2,random_state=42)

x_placeholder = tf.placeholder(tf.float32,[None,300,300,3])
y_placeholder = tf.placeholder(tf.float32,[None,300,300,1])

#encoder
conv1 = tf.layers.conv2d(x_placeholder,64,3,activation = tf.nn.relu,padding="SAME")
conv1 = tf.layers.conv2d(conv1,64,3,activation = tf.nn.relu,padding="SAME")
crop1 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv1)
conv1 = tf.layers.max_pooling2d(conv1,2,2,padding = "SAME")


conv2 = tf.layers.conv2d(conv1,128,3,activation = tf.nn.relu,padding="SAME")
conv2 = tf.layers.conv2d(conv2,128,3,activation = tf.nn.relu,padding="SAME")
crop2 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv2)
conv2 = tf.layers.max_pooling2d(conv2,2,2,padding = "SAME")

conv3 = tf.layers.conv2d(conv2,256,3,activation = tf.nn.relu,padding="SAME")
conv3 = tf.layers.conv2d(conv3,256,3,activation = tf.nn.relu,padding="SAME")
crop3 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv3)
conv3 = tf.layers.max_pooling2d(conv3,2,2,padding = "SAME")

conv4 = tf.layers.conv2d(conv3,512,3,activation = tf.nn.relu,padding="SAME")
conv4 = tf.layers.conv2d(conv4,512,3,activation = tf.nn.relu,padding="SAME")
crop4 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv4)
conv4 = tf.layers.max_pooling2d(conv4,2,2,padding = "SAME")

conv5 = tf.layers.conv2d(conv4,1024,3,activation = tf.nn.relu,padding="SAME")
conv5 = tf.layers.conv2d(conv5,1024,3,activation = tf.nn.relu,padding="SAME")


#decoder
conv6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
conv6_1 = tf.layers.conv2d(conv6, 512, 3, activation=tf.nn.relu, padding="SAME")
conv6 = tf.concat([crop4, conv6_1], axis=3)
conv6 = tf.layers.conv2d(conv6,512,3,activation = tf.nn.relu,padding="SAME")
conv6 = tf.layers.conv2d(conv6,512,3,activation = tf.nn.relu,padding="SAME")

conv7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
conv7_1 = tf.layers.conv2d(conv7, 256, 3, activation=tf.nn.relu, padding="SAME")
crop7_1 = tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)))(conv7_1)
conv7 = tf.concat([crop3, crop7_1], axis=3)
conv7 = tf.layers.conv2d(conv7,256,3,activation = tf.nn.relu,padding="SAME")
conv7 = tf.layers.conv2d(conv7,256,3,activation = tf.nn.relu,padding="SAME")

conv8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
conv8_1 = tf.layers.conv2d(conv8, 128, 2, activation=tf.nn.relu, padding="SAME")
conv8 = tf.concat([crop2, conv8_1], axis=3)
conv8 = tf.layers.conv2d(conv8,128,3,activation = tf.nn.relu,padding="SAME")
conv8 = tf.layers.conv2d(conv8,128,3,activation = tf.nn.relu,padding="SAME")

conv9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
conv9_1 = tf.layers.conv2d(conv9, 64, 2, activation=tf.nn.relu, padding="SAME")
conv9 = tf.concat([crop1, conv9_1], axis=3)
conv9 = tf.layers.conv2d(conv9,64,3,activation = tf.nn.relu,padding="SAME")
conv9 = tf.layers.conv2d(conv9,64,3,activation = tf.nn.relu,padding="SAME")
conv9 = tf.layers.conv2d(conv9,64,3,activation = tf.nn.relu,padding="SAME")
conv9 = tf.layers.conv2d(conv9,10,3,activation = tf.nn.relu,padding="SAME")

conv10 = tf.layers.conv2d(conv9, 1, 1,padding="SAME")

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

sess.run([tf.shape(conv10)],feed_dict={x_placeholder:X_train[0:3]})

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_placeholder,logits=conv10)
loss = tf.math.reduce_mean(loss)

sess.run([tf.shape(loss)],feed_dict={x_placeholder:X_train[0:3],y_placeholder:y_train[0:3]})

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)

train_op = optimizer.minimize(loss)

batch_size = 4
def random_batch(x_train,y_train,batch_size):
  rnd_indices = np.random.randint(0,len(x_train),batch_size) 
  x_batch = x_train[rnd_indices]
  y_batch = y_train[rnd_indices]
  return x_batch,y_batch

init = tf.global_variables_initializer()
sess.run(init)

num_steps =200
for step in range(1, num_steps+1):
    x_batch, y_batch = random_batch(X_train, y_train, batch_size)
    sess.run(train_op, feed_dict={x_placeholder:x_batch,y_placeholder:y_batch})
    if step % 10 == 0:
        loss_train = sess.run( loss, feed_dict={x_placeholder: x_batch,y_placeholder: y_batch})
        x_batch, y_batch = random_batch(X_validation, y_validation, batch_size)
        loss_validation = sess.run( loss, feed_dict={x_placeholder: x_batch,y_placeholder: y_batch})
        print('Step:',step, ', loss_train:',loss_train,"loss_validation:",loss_validation)
print("Optimization Finished!")

x_batch, y_batch = random_batch(X_validation, y_validation, 10)
loss_validation = sess.run( loss, feed_dict={x_placeholder: x_batch,y_placeholder: y_batch})
print(loss_validation)

i=12
img=X_validation[i]
pred_value=sess.run( conv10, feed_dict={x_placeholder:[img]})[0]
img=np.array((img*128+128),dtype=np.uint8)
import copy

img_copy=copy.deepcopy(img)
import matplotlib.pyplot as plt
label=np.array(y_validation[i]*255,dtype=np.uint8)

for i in range(300):
  for j in range(300):
    if label[i,j]>100:
      img_copy[i,j,1]=255
plt.imshow(img_copy)
plt.show()

class_predict=1 / (1 + np.exp(-pred_value))
mask=cv2.inRange(class_predict,0.2,1)

for i in range(300):
  for j in range(300):
    if mask[i,j]>100:
      img[i,j,0]=255
plt.imshow(img)
plt.show()


# -*- coding: utf-8 -*-
"""Attacking AES-ECB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PC7u4yD0s2VciTzFnEuJU_82reE4djpa
"""

# !pip install pycrypto
# !pip install pycryptodome
# !pip install keras
# !pip install tensorflow

from Crypto.Cipher import AES
import os 
import pandas as pd
import numpy as np


import binascii
from base64 import b64encode, b64decode
from Crypto import Random
import pdb
from Crypto.Util import Padding
from keras.models import Sequential
from keras import layers
import tensorflow as tf


data1 =  pd.read_csv('Train.csv')
data2 =  pd.read_csv('Test.csv')
data3 =  pd.read_csv('Valid.csv')


df = pd.DataFrame()
df['Plaintext'] = data1['text']
df2 = pd.DataFrame()
df2['Plaintext'] = data2['text']
df3 = pd.DataFrame()
df3['Plaintext'] = data3['text']
df = df.append(df2, ignore_index=True)
df = df.append(df3, ignore_index=True)
df['Not padded Plaintext'] = '' 
df['Key16'] = ''
df['Ciphertext16'] = ''
df['Key24'] = ''
df['Ciphertext24'] = ''
df['Key32'] = ''
df['Ciphertext32'] = ''
print(df)

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
with tf.device('/device:GPU:0'):
  #creating random keys 16,24,32
  for i in df.index:
    random_key = os.urandom(16)
    df.iat[i, 2] = random_key

  for i in df.index:
    random_key = os.urandom(24)
    df.iat[i, 4] = random_key

  for i in df.index:
    random_key = os.urandom(32)
    df.iat[i, 6] = random_key

def encryption(plain_text, key):
  #text = Padding.pad(plain_text, AES.block_size, style = 'pkcs7')
  cipher = AES.new(key, mode=1)
  encrypted_text = cipher.encrypt(plain_text)
  return encrypted_text

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
with tf.device('/device:GPU:0'):
  #removing chars from texts with len !%16==0
  for i in df.index:
    text = str.encode(df.iloc[i]['Plaintext'])
    if len(text) % 16 != 0:
      (df.at[i,'Not padded Plaintext']) = text[:len(text)-(len(text)%16)]
    else:
      (df.at[i,'Not padded Plaintext']) = text

print(df['Not padded Plaintext'])

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
with tf.device('/device:GPU:0'):
    #encrypting
    for i in df.index:
      text = df.iloc[i]['Not padded Plaintext']
      key = df.iloc[i]['Key16']
      e = encryption(text,key)
      df.at[i,'Ciphertext16']=e
    

    for i in df.index:
      text = df.iloc[i]['Not padded Plaintext']
      key = df.iloc[i]['Key24']
      e = encryption(text,key)
      df.at[i,'Ciphertext24']=e
      

    for i in df.index:
      text = df.iloc[i]['Not padded Plaintext']
      key = df.iloc[i]['Key32']
      e = encryption(text,key)
      df.at[i,'Ciphertext32']=e

print(df['Ciphertext16'])
print(df['Ciphertext24'])
print(df['Ciphertext32'])

# byte to bit of ciphertext
df['BitCiphertext16'] = ''
for i in df.index:
  df.at[i,'BitCiphertext16'] = ''.join(format(j, 'b') for j in df.iloc[i]['Ciphertext16']) 

print(df['BitCiphertext16'])

#byte to bit plaintext
df['BitPlaintext'] = ''
for i in df.index:
  df.at[i,'BitPlaintext'] = ''.join(format(j, 'b') for j in df.iloc[i]['Not padded Plaintext']) 

print(df['BitPlaintext'])

# len(bits) for training up to %128==0
df['TrainPlain'] = ''
df['Train16'] = ''

for i in df.index:
  text = df.iloc[i]['BitPlaintext']
  if len(text) % 128 != 0:
    (df.at[i,'TrainPlain']) = text[:len(text)-(len(text)%128)]
  else:
    (df.at[i,'TrainPlain']) = text

for i in df.index:
  cipher = df.iloc[i]['BitCiphertext16']
  if len(cipher) % 128 != 0:
    (df.at[i,'Train16']) = cipher[:len(cipher)-(len(cipher)%128)]
  else:
    (df.at[i,'Train16']) = cipher

#len(bits plain) == len(bits cipher)

for i in df.index:
  plain = df.iloc[i]['TrainPlain']
  cipher = df.iloc[i]['Train16']
  if(len(plain) > len(cipher)):
    df.at[i,'TrainPlain'] = plain[:len(cipher)]
  else:
    df.at[i,'Train16'] = plain[:len(plain)]

# count = 0
# for i in df.index:
#   count = count + (len(df.at[i,'TrainPlain'])/128)
# print(count)

# # nxm n=128 m=3425019
# plain_data = np.empty((128,3425019))
# cipher16_data =  np.empty((128,3425019))

# plain = df.iloc[0]['TrainPlain']
# m = int(len(plain)/128)
# for i in range(0,m-1):
#   for n in range(0,127):
#     for bit in plain:
#       plain_data[i][n] = int(bit)

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
number_columns = 128
device_name = tf.test.gpu_device_name()
with tf.device('/device:GPU:0'):
  for i in range(0,10000):
    sample_string = df.iloc[i]['TrainPlain']  
    l = [list(sample_string[i:i+number_columns]) for i in range(0, len(sample_string), number_columns)]
    if(i==0):
      matrix = np.array([s if len(s) == number_columns else s+[None]*(number_columns-len(s)) for s in l])
      matrix2 = np.array([s if len(s) == number_columns else s+[None]*(number_columns-len(s)) for s in l])
      final = np.concatenate((matrix, matrix2))
    else:
      matrix2 = np.array([s if len(s) == number_columns else s+[None]*(number_columns-len(s)) for s in l])
      final = np.concatenate((final, matrix2))
    print('Done', i)

final

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
number_columns = 128
device_name = tf.test.gpu_device_name()
with tf.device('/device:GPU:0'):
  for i in df.index:
    sample_string = df.iloc[i]['Train16']  
    l = [list(sample_string[i:i+number_columns]) for i in range(0, len(sample_string), number_columns)]
    if(i==0):
      matrix = np.array([s if len(s) == number_columns else s+[None]*(number_columns-len(s)) for s in l])
      matrix2 = np.array([s if len(s) == number_columns else s+[None]*(number_columns-len(s)) for s in l])
      final2 = np.concatenate((matrix, matrix2))
    else:
      matrix2 = np.array([s if len(s) == number_columns else s+[None]*(number_columns-len(s)) for s in l])
      final2 = np.concatenate((final2, matrix2))
    print('Done', i)
    if(i==10000):
      break

final2

#ML approach 
from sklearn.model_selection import train_test_split
# TrainPlain 10 - plaintext in bits
# Train16 11 - ciphertext in bits
# X = df.iloc[:,10:11].values
# y = df.iloc[:,9:10].values

X = final2
y = final

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

input_dim = X_train.shape[1]  # Number of features


model = Sequential()
model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(128, activation='sigmoid'))

X_train.shape #(40000,1)
X_test.shape #(10000,1)
y_train.shape
y_test.shape

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=100,verbose=False,validation_data=(X_test,y_test),batch_size=5000)

# from keras.layers import Input, Dense
# from keras.models import Model

# # I assume that x is the array containing the training data
# # the shape of x should be (num_samples, 4)
# # The array containing the test data is named y and is 
# # one-hot encoded with a shape of (num_samples, num_classes)
# # num_samples is the number of samples in your training set
# # num_classes is the number of classes you have
# # e.g. is a binary classification problem num_classes=2

# # First, we'll define the architecture of the network
# inp = Input(shape=(1,)) # you have 1 features
# hidden = Dense(2, activation='sigmoid')(inp)  # 10 neurons in your hidden layer
# out = Dense(activation='softmax', units=1)(hidden)  

# # Create the model
# model = Model(inputs=[inp], outputs=[out])

# # Compile the model and define the loss function and optimizer
# model.compile(loss='mse', optimizer='adam', 
#               metrics=['accuracy'])
# # feel free to change these to suit your needs

# # Train the model
# model.fit(X, y, epochs=100, batch_size=5000)
# # train the model for 10 epochs with a batch size of 512

size = 0
for i in df.index:
  text = len((df.iloc[i]['Not padded Plaintext']).split())
  size += text

avg = size / len(df.index)

print(avg)

imput_matrix = np.array()
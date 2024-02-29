#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[7]:


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape


# In[8]:


df = pd.read_csv('sudoku.csv.Zip')
df.head()


# In[9]:


que = df['quizzes'].values
soln = df['solutions']


# In[10]:


feat = []
label = []

for i in que:

    x = np.array([int(j) for j in i]).reshape((9,9,1))
    feat.append(x)

feat = np.array(feat)
feat = feat/9
feat -= .5    

for i in soln:

    x = np.array([int(j) for j in i]).reshape((81,1)) - 1
    label.append(x)   

label = np.array(label)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(feat,label, test_size=0.33, random_state=42)


# In[12]:


def get_model():

    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    
    return model


# In[13]:


model = get_model()


# In[14]:


adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

model.fit(X_train, y_train, batch_size=32, epochs=1)


# In[16]:


def denorm(a):
    
    return (a+.5)*9


# In[17]:


def norm(a):
    
    return (a/9)-.5


# In[18]:


import copy


# In[19]:


def inference_sudoku(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = copy.copy(sample)
    
    while(1):
        
        #predicting values
        out = model.predict(feat.reshape((1,9,9,1)))  
        out = out.squeeze()

        #getting predicted values
        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        #getting probablity of each values
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        #creating mask for blank values
        feat = denorm(feat).reshape((9,9))
        #i.e it will give you a 2d array with True/1 and False/0 where 0 is found and where 0 is not found respectively
        mask = (feat==0)
     
        #if there are no 0 values left than break
        if(mask.sum()==0):
            break
            
        #getting probablities of values where 0 is present that is our blank values we need to fill
        prob_new = prob*mask
    
        #getting highest probablity index
        ind = np.argmax(prob_new)
        #getting row and col 
        x, y = (ind//9), (ind%9)
        
        #getting predicted value at that cell
        val = pred[x][y]
        #assigning that value
        feat[x][y] = val
        #again passing this sudoku with one value added to model to get next most confident value
        feat = norm(feat)
    
    return pred


# In[20]:


def test_accuracy(feats, labels):
    
    correct = 0
    
    for i,feat in enumerate(feats):
        
        pred = inference_sudoku(feat)
        
        true = labels[i].reshape((9,9))+1
        
        if(abs(true - pred).sum()==0):
            correct += 1
        
    print(correct/feats.shape[0])


# In[ ]:


test_accuracy(X_test[:100], y_test[:100])


# In[ ]:


def solve_sudoku(game):
    
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


# In[ ]:


game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''

game = solve_sudoku(game)

print('solved puzzle:\n')
# print(game)


# In[ ]:


for i in game:
    print(i)


# In[ ]:


np.sum(game, axis=1)


# In[ ]:





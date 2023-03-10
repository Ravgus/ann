# -*- coding: utf-8 -*-
"""Mega_Case_Study.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qdcYVl1DEvt6oB5qxaQWoF6kWWz2BJGB

#Mega Case Study - Making a hybrid Deep Learning Model

#Part 1 - SOM

##Install MiniSom Package
"""

#!pip install MiniSom

"""## Importing the libraries

"""

import numpy as np
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

"""## Feature Scaling

"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM

"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results

"""

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding the frauds

"""

mappings = som.win_map(X)
potential_frauds = np.where(som.distance_map().T>0.9)
zipped_potential_frauds = [tuple(t) for t in zip(potential_frauds[1], potential_frauds[0])]

frauds = []

for coords in zipped_potential_frauds:
    fraud = mappings[coords]
    if len(fraud) != 0:
        frauds.append(fraud)
        
frauds = np.concatenate(frauds, axis = 0)
frauds = sc.inverse_transform(frauds)

"""##Printing the Fraunch Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))

"""#Part 2 - Going from Unsupervised to Supervised Deep Learning

##Create Matrix of Features
"""

customers = dataset.iloc[:, 1:].values

"""## Create Dependent Variable"""

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1

"""#Part 3 - ANN

### Feature Scaling
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

"""## Building the ANN

##Import Tensorflow
"""

import tensorflow as tf

"""## Initializing the ANN"""

ann = tf.keras.models.Sequential()

"""##Adding the input layer and the first hidden layer"""

ann.add(tf.keras.layers.Dense(units=2, activation='relu'))

"""## Adding the output layer"""

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Training the ANN

## Compiling the ANN
"""

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""## Training the ANN on the Training set"""

ann.fit(customers, is_fraud, batch_size = 1, epochs = 10)

"""## Predicting test set results"""

y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]

print(y_pred)
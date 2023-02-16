# -*- coding: utf-8 -*-
"""Final_ML_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I7xzbmHsQMylznLf-_LmBsQpuvSKGozJ

# **Imports**
"""

import io
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['Star_Dataset.csv']))
# Dataset is now stored in a Pandas Dataframe

"""# **Pre Processing**

Data can sometimes be too complex or include data that is not exactly continuous or linear, to fix this we can apply some simple feature engineering to make sure class features are separated well enough to make predictions. Methods include scaling, x = (xi-u)/s where u is the mean and s is the standard deviation, basically standarizing the features. Also one hot encoding to make use of binary or string attributes depending on their frequency.
"""

def preprocess_inputs(df):
    #same color classes had different names (white/White) so we create a dicitonary to standarize the classes.
    color_mapping = {
        'Blue ': 'Blue',
        'Blue white': 'Blue White',
        'Blue-white': 'Blue White',
        'Blue white ': 'Blue White',
        'Blue-White': 'Blue White',
        'white': 'White',
        'yellow-white': 'Yellowish White',
        'White-Yellow': 'Yellowish White',
        'yellowish': 'Yellowish'
    }
    #replacing the incorrect lables to the standarized version
  
    df['Star color'] = df['Star color'].replace(color_mapping)
    
    # One-hot encoding
    df = pd.concat([df, pd.get_dummies(df['Star color'], prefix='Star_Color')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Spectral Class'], prefix='Spectrum')], axis=1)
    df = df.drop(['Star color', 'Spectral Class'], axis=1)
    
    # Split X and y
    
    X = df.drop('Star type', axis=1)
    y = df['Star type']
    
    return X, y



X, Y = preprocess_inputs(df)
#print(X)
print(X.shape)
X = X.values
Y = Y.values

star_type = {
        0: 'Brown Dwarf',
        1: 'Red Dwarf',
        2: 'White Dwarf',
        3: 'Main Sequence',
        4: 'Supergiant',
        5: 'Hypergiant',
    }

sorted_labels = ['Brown Dwarf','Red Dwarf','White Dwarf', 'Main Sequence', 'Supergiant','Hypergiant']

"""#**Guassian NB**

Very fast and simple model that can be traced by hand. it is easily taught and does not require great knowledge of calculus. Explaining its simple formula can help understand how data affects predictions



![GNB.jpg](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAeAB4AAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCACJAgMDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoooo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)
"""

kf = KFold(n_splits = (10), shuffle = True)

accuracyAverage = 0
gnb = GaussianNB()
sc = StandardScaler()
#list to append actual and predicted classes for KFold
actual_classes = np.empty([0], dtype=int)
predicted_classes = np.empty([0], dtype=int)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  Y_train, Y_test = Y[train_index], Y[test_index]
  #Scale X
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  #fit gaussian model
  model=gnb.fit(X_train, Y_train)
  y_pred = model.predict(X_test)
  #get current run actual and predicted classes
  actual_classes = np.append(actual_classes, Y_test)
  predicted_classes = np.append(predicted_classes, y_pred)

  #sum accuracy
  accuracyAverage += accuracy_score(Y_test, y_pred)



matrix = confusion_matrix(actual_classes, predicted_classes)

#print accuracy average 
print("Accuracy:", accuracyAverage/kf.get_n_splits(X), "\n")

plt.figure(figsize=(10,6))
sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('GNB Confusion Matrix')

plt.show()

"""# **Logistic Regression**

Another simple model that adds a layer of complexity to models like linear regression by using drawn from a function that obtains probabilities in a range between 0 and 1

![LogReg_1.png](data:image/png;base64,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)
"""

kf = KFold(n_splits = (10), shuffle = True)
accuracyAverage = 0
lg = LogisticRegression()
sc = StandardScaler()

#list to append actual and predicted classes for KFold
actual_classes = np.empty([0], dtype=int)
predicted_classes = np.empty([0], dtype=int)

for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  Y_train, Y_test = Y[train_index], Y[test_index]
  #scale X
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  #fit model
  model=lg.fit(X_train, Y_train)
  y_pred = model.predict(X_test)
  #get current run actual and predicted classes
  actual_classes = np.append(actual_classes, Y_test)
  predicted_classes = np.append(predicted_classes, y_pred)

  accuracyAverage += accuracy_score(Y_test, y_pred)

matrix = confusion_matrix(actual_classes, predicted_classes)
print("Accuracy:", accuracyAverage/kf.get_n_splits(X), "\n")

plt.figure(figsize=(10,6))
sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('LogReg Confusion Matrix')

plt.show()

"""# **FF Neural Network**

A more complex approach to modelling, this approach was included to demonstrate how great complexity is not always necessary for simple data but to also show its non-determenistic ways and less need for validation techniques
"""

import tensorflow as tf
from tensorflow import keras
from keras import metrics
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, shuffle=True, random_state=0)

#shuffling order of training examples

indices = np.arange(X_train.shape[0])
shuffled_indices = np.random.permutation(indices)

X_train = X_train[shuffled_indices]
Y_train = Y_train[shuffled_indices]

#Scaling X
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def build_ffnn_softmax_model(input_shape, n_classes, learning_rate):
    #reset seed and session
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    #FF model
    model = keras.Sequential()
    #add input layer, 1 hidden layer and output layer
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(units = 80, activation= 'relu'))
    model.add(keras.layers.Dense(
        units = n_classes, activation= 'softmax'
    ))
    #optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer = optimizer,
                 metrics = ['accuracy'])
    
    return model


#build model
ffnn_softmax_model = build_ffnn_softmax_model(X_train[0].shape, X_train.shape[1], 0.3)
#fit X and Y train
ffnn_softmax_model.fit(
  x = X_train,
  y = Y_train,
  epochs=5,
  batch_size=64,
  validation_split=0.1,
  verbose=1)
#get predictions from model
ffnn_softmax_test_predictions = np.argmax(ffnn_softmax_model.predict(X_test), axis=-1)
#print accuracy
print("Accuracy:", accuracy_score(Y_test, ffnn_softmax_test_predictions), "\n")
#get confussion matrix
matrix = confusion_matrix(Y_test, ffnn_softmax_test_predictions)

plt.figure(figsize=(10,6))
sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('FFNN Confusion Matrix')

plt.show()
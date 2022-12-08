# TensorFlow and TensorBoard with Regularization



## Purpose

The purpose of this lab is threefold.  

1.   to review using `TensorFlow` and `TensorBoard` for modeling and evaluation with neural networks
2.   to review using data science pipelines and cross-validation with neural networks
3.   to review using `TensorFlow` for neural network regularization

We'll be continuting our investigation of the canonical [Titanic Data Set](https://www.kaggle.com/competitions/titanic/overview) that we began [previously](https://github.com/learn-co-curriculum/enterprise-paired-nn-eval).

## The Titanic

### The Titanic and it's data



RMS Titanic was a British passenger liner built by Harland and Wolf and operated by the White Star Line. It sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton, England to New York City, USA.

Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. 

Though there were about 2,224 passengers and crew members, we are given data of about 1,300 passengers. Out of these 1,300 passengers details, about 900 data is used for training purpose and remaining 400 is used for test purpose. The test data has had the survived column removed and we'll use neural networks to predict whether the passengers in the test data survived or not. Both training and test data are not perfectly clean as we'll see.

Below is a picture of the Titanic Museum in Belfast, Northern Ireland.


```python
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://upload.wikimedia.org/wikipedia/commons/c/c0/Titanic_Belfast_HDR.jpg", width=400, height=400)
```




<img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Titanic_Belfast_HDR.jpg" width="400" height="400"/>



### Data Dictionary

*   *Survival* : 0 = No, 1 = Yes
*   *Pclass* : A proxy for socio-economic status (SES)
  *   1st = Upper
  *   2nd = Middle
  *   3rd = Lower
*   *sibsp* : The number of siblings / spouses aboard the Titanic
  *   Sibling = brother, sister, stepbrother, stepsister
  *   Spouse = husband, wife (mistresses and fiancés were ignored)
*   *parch* : The # of parents / children aboard the Titanic
  *   Parent = mother, father
  *   Child = daughter, son, stepdaughter, stepson
  *   Some children travelled only with a nanny, therefore *parch*=0 for them.
*   *Ticket* : Ticket number
*   *Fare* : Passenger fare (British pounds)
*   *Cabin* : Cabin number embarked
*   *Embarked* : Port of Embarkation
  *   C = Cherbourg (now Cherbourg-en-Cotentin), France
  *   Q = Queenstown (now Cobh), Ireland
  *   S = Southampton, England
*   *Name*, *Sex*, *Age* (years) are all self-explanatory

## Libraries and the Data



### Importing libraries


```python
# Load the germane libraries

import pandas as pd
import numpy as np
import seaborn as sns 
from pandas._libs.tslibs import timestamps
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras 
from keras import models
from sklearn.impute import SimpleImputer
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.losses import binary_crossentropy
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier

# Load the TensorBoard notebook extension and related libraries
%load_ext tensorboard
import datetime
```

### Loading the data


```python
# Load the data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# We need to do this for when we mamke our predictions from the test data at the end
ids = test[['PassengerId']]
```

## EDA and Preprocessing

### Exploratory Data Analysis

You have already performed EDA on this data set. Look back on what you did before or see [here](https://github.com/learn-co-curriculum/enterprise-paired-nn-eval).

Of course, feel free to re-run what you have done before or try out some other EDA as you find useful.

### Preprocessing

Let's do the same prepricessing as before.


```python
# Performing preprocessing on the train and test data will be more effecient if we combine the two date sets.
combined = pd.concat([train, test], axis=0, sort=False)

#Age column
combined['Age'].fillna(combined['Age'].median(),inplace=True) # Age

# Embarked column
combined['Embarked'].fillna(combined['Embarked'].value_counts().index[0], inplace=True) # Embarked
combined['Fare'].fillna(combined['Fare'].median(),inplace=True)

# Class column
d = {1:'1st',2:'2nd',3:'3rd'} #Pclass
combined['Pclass'] = combined['Pclass'].map(d) #Pclass

# Making Age into adult (1) and child (0)
combined['Child'] = combined['Age'].apply(lambda age: 1 if age>=18 else 0) 

# Break up the string that has the title and names
combined['Title'] = combined['Name'].str.split('.').str.get(0)  # output : 'Futrelle, Mrs'
combined['Title'] = combined['Title'].str.split(',').str.get(1) # output : 'Mrs '
combined['Title'] = combined['Title'].str.strip()               # output : 'Mrs'
combined.groupby('Title').count()

# Replace the French titles with Enlgish
french_titles = ['Don', 'Dona', 'Mme', 'Ms', 'Mra','Mlle']
english_titles = ['Mr', 'Mrs','Mrs','Mrs','Mrs','Miss']
for i in range(len(french_titles)):
    for j in range(len(english_titles)):
        if i == j:
            combined['Title'] = combined['Title'].str.replace(french_titles[i],english_titles[j])

# Seperate the titles into "major" and "others", the latter would be, e.g., Reverend
major_titles = ['Mr','Mrs','Miss','Master']
combined['Title'] = combined['Title'].apply(lambda title: title if title in major_titles else 'Others')

#Dropping the Irrelevant Columns
combined.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace=True)

# Getting Dummy Variables and Dropping the Original Categorical Variables
categorical_vars = combined[['Pclass','Sex','Embarked','Title','Child']] # Get Dummies of Categorical Variables
dummies = pd.get_dummies(categorical_vars,drop_first=True)
combined = combined.drop(['Pclass','Sex','Embarked','Title','Child'],axis=1)
combined = pd.concat([combined, dummies],axis=1)

# Separating the data back into train and test sets
test = combined[combined['Survived'].isnull()].drop(['Survived'],axis=1)
train = combined[combined['Survived'].notnull()]

# Training
X_train = train.drop(['Survived'],1)
y_train = train['Survived']

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
test = sc.fit_transform(test)
```

    <ipython-input-4-15c4eb6bfb15>:37: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      combined.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace=True)
    <ipython-input-4-15c4eb6bfb15>:50: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      X_train = train.drop(['Survived'],1)


## Neural Network Model

### Building the model

#### Define the model as a pipeline

Let's use the data science pipeline for our neural network model.

As you are now using regularization to guard against high variance, i.e. overfitting the data, in the definition of the model below include *dropout* and/or *l2* regularization. Also, feel free to experiment with different activation functions.


```python
# It will help to define our model in terms of a pipeline
def build_classifier(optimizer):
# insert Sequential and layers here

    return classifier
```


```python
# __SOLUTION__

# It will help to define our model in terms of a pipeline
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=10,kernel_regularizer='l2',activation='relu',input_dim=14))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units=128,kernel_regularizer='l2',activation='relu'))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units=1,kernel_regularizer='l2',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
```

#### Use grid search to find help you tune the parameters

You can play with optimizers, epochs, and batch sizes. The ones that we're suggesting are not necessarily the best.


```python
# Grid Search
classifier = KerasClassifier(build_fn = build_classifier)
param_grid = dict(optimizer = ['Adam'],
                  epochs=[10, 20, 50],
                  batch_size=[16, 25, 32])
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy')
grid_result = grid.fit(X_train, y_train)
best_parameters = grid.best_params_
best_accuracy = grid.best_score_
```

    <ipython-input-7-182bbc6778dc>:2: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      classifier = KerasClassifier(build_fn = build_classifier)


    Epoch 1/10
    45/45 [==============================] - 1s 2ms/step - loss: 0.9468 - accuracy: 0.6742
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.8200 - accuracy: 0.7472
    Epoch 3/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7368 - accuracy: 0.7598
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7038 - accuracy: 0.7669
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6608 - accuracy: 0.7767
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6366 - accuracy: 0.7837
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6078 - accuracy: 0.8006
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5862 - accuracy: 0.8076
    Epoch 9/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5742 - accuracy: 0.8104
    Epoch 10/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5587 - accuracy: 0.8076
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 2ms/step - loss: 0.9449 - accuracy: 0.6942
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.8032 - accuracy: 0.7714
    Epoch 3/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7196 - accuracy: 0.7812
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6798 - accuracy: 0.7882
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6445 - accuracy: 0.8065
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6314 - accuracy: 0.7924
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6127 - accuracy: 0.8008
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6088 - accuracy: 0.7728
    Epoch 9/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5869 - accuracy: 0.7966
    Epoch 10/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5654 - accuracy: 0.7980
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 2ms/step - loss: 0.9059 - accuracy: 0.6718
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7825 - accuracy: 0.7574
    Epoch 3/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7199 - accuracy: 0.7896
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6758 - accuracy: 0.7994
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6327 - accuracy: 0.8135
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6013 - accuracy: 0.8261
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5896 - accuracy: 0.8135
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5795 - accuracy: 0.8022
    Epoch 9/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5695 - accuracy: 0.8191
    Epoch 10/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5536 - accuracy: 0.8205
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 2ms/step - loss: 0.9120 - accuracy: 0.6297
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7942 - accuracy: 0.7013
    Epoch 3/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7263 - accuracy: 0.7602
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6755 - accuracy: 0.7994
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6364 - accuracy: 0.8079
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6148 - accuracy: 0.8163
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5864 - accuracy: 0.8205
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5666 - accuracy: 0.8219
    Epoch 9/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5528 - accuracy: 0.8289
    Epoch 10/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5417 - accuracy: 0.8289
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 2ms/step - loss: 0.9520 - accuracy: 0.5540
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.8130 - accuracy: 0.7251
    Epoch 3/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7443 - accuracy: 0.7532
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6978 - accuracy: 0.7770
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6607 - accuracy: 0.7784
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6334 - accuracy: 0.7938
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6164 - accuracy: 0.7966
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6057 - accuracy: 0.7952
    Epoch 9/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5902 - accuracy: 0.8036
    Epoch 10/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5756 - accuracy: 0.7994
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9262 - accuracy: 0.6587
    Epoch 2/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.8011 - accuracy: 0.7416
    Epoch 3/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7226 - accuracy: 0.7528
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6856 - accuracy: 0.7795
    Epoch 5/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6405 - accuracy: 0.7697
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6118 - accuracy: 0.7935
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5999 - accuracy: 0.7921
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5851 - accuracy: 0.7963
    Epoch 9/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5612 - accuracy: 0.8034
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5481 - accuracy: 0.8062
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5404 - accuracy: 0.8146
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5265 - accuracy: 0.8174
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5255 - accuracy: 0.8188
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5150 - accuracy: 0.8258
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5176 - accuracy: 0.8188
    Epoch 16/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5156 - accuracy: 0.8202
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5057 - accuracy: 0.8230
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4943 - accuracy: 0.8287
    Epoch 19/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5008 - accuracy: 0.8272
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4877 - accuracy: 0.8287
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9334 - accuracy: 0.6536
    Epoch 2/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.8179 - accuracy: 0.6816
    Epoch 3/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7351 - accuracy: 0.7588
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6956 - accuracy: 0.7756
    Epoch 5/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6434 - accuracy: 0.7896
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6224 - accuracy: 0.7882
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6005 - accuracy: 0.7966
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5672 - accuracy: 0.8107
    Epoch 9/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5471 - accuracy: 0.8191
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5344 - accuracy: 0.8233
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5235 - accuracy: 0.8289
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5257 - accuracy: 0.8219
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5251 - accuracy: 0.8121
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5209 - accuracy: 0.8247
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5138 - accuracy: 0.8177
    Epoch 16/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4998 - accuracy: 0.8261
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4982 - accuracy: 0.8247
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4914 - accuracy: 0.8303
    Epoch 19/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4952 - accuracy: 0.8191
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4986 - accuracy: 0.8261
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 5ms/step - loss: 0.9254 - accuracy: 0.6480
    Epoch 2/20
    45/45 [==============================] - 0s 5ms/step - loss: 0.8116 - accuracy: 0.7181
    Epoch 3/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.7253 - accuracy: 0.7518
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6842 - accuracy: 0.7798
    Epoch 5/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6476 - accuracy: 0.7854
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6131 - accuracy: 0.7938
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5949 - accuracy: 0.8079
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5739 - accuracy: 0.8065
    Epoch 9/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5590 - accuracy: 0.8163
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5422 - accuracy: 0.8149
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5493 - accuracy: 0.7994
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5243 - accuracy: 0.8163
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5131 - accuracy: 0.8275
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5070 - accuracy: 0.8205
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5028 - accuracy: 0.8247
    Epoch 16/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5201 - accuracy: 0.8163
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5090 - accuracy: 0.8149
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4890 - accuracy: 0.8345
    Epoch 19/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5076 - accuracy: 0.8135
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4908 - accuracy: 0.8345
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9523 - accuracy: 0.6830
    Epoch 2/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7832 - accuracy: 0.7826
    Epoch 3/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7139 - accuracy: 0.8065
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6569 - accuracy: 0.8149
    Epoch 5/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6433 - accuracy: 0.8149
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6291 - accuracy: 0.8065
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5986 - accuracy: 0.8205
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5783 - accuracy: 0.8275
    Epoch 9/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5578 - accuracy: 0.8135
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5443 - accuracy: 0.8233
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5457 - accuracy: 0.8191
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5315 - accuracy: 0.8191
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5288 - accuracy: 0.8163
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5166 - accuracy: 0.8233
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5231 - accuracy: 0.8289
    Epoch 16/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5081 - accuracy: 0.8303
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5078 - accuracy: 0.8331
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4947 - accuracy: 0.8317
    Epoch 19/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5067 - accuracy: 0.8331
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4952 - accuracy: 0.8331
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9395 - accuracy: 0.6592
    Epoch 2/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.8312 - accuracy: 0.7167
    Epoch 3/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7467 - accuracy: 0.7518
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6967 - accuracy: 0.7756
    Epoch 5/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6767 - accuracy: 0.7714
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6491 - accuracy: 0.7882
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6244 - accuracy: 0.7784
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6025 - accuracy: 0.7854
    Epoch 9/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5872 - accuracy: 0.7840
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5876 - accuracy: 0.7868
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5704 - accuracy: 0.7994
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5602 - accuracy: 0.7938
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5438 - accuracy: 0.8050
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5435 - accuracy: 0.8121
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5417 - accuracy: 0.8022
    Epoch 16/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5295 - accuracy: 0.8149
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5179 - accuracy: 0.8065
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5238 - accuracy: 0.8205
    Epoch 19/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5184 - accuracy: 0.8093
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5265 - accuracy: 0.8135
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9264 - accuracy: 0.6657
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7977 - accuracy: 0.7472
    Epoch 3/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7290 - accuracy: 0.7612
    Epoch 4/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6815 - accuracy: 0.7781
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6454 - accuracy: 0.7837
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6199 - accuracy: 0.7837
    Epoch 7/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5932 - accuracy: 0.7907
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5799 - accuracy: 0.8076
    Epoch 9/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5730 - accuracy: 0.8160
    Epoch 10/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5561 - accuracy: 0.8104
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5541 - accuracy: 0.7992
    Epoch 12/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5347 - accuracy: 0.8104
    Epoch 13/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5355 - accuracy: 0.8104
    Epoch 14/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5243 - accuracy: 0.8258
    Epoch 15/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5267 - accuracy: 0.8048
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5147 - accuracy: 0.8174
    Epoch 17/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4958 - accuracy: 0.8272
    Epoch 18/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5010 - accuracy: 0.8357
    Epoch 19/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4992 - accuracy: 0.8357
    Epoch 20/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5076 - accuracy: 0.8160
    Epoch 21/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4823 - accuracy: 0.8258
    Epoch 22/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5045 - accuracy: 0.8160
    Epoch 23/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4977 - accuracy: 0.8146
    Epoch 24/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4855 - accuracy: 0.8357
    Epoch 25/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4951 - accuracy: 0.8315
    Epoch 26/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4766 - accuracy: 0.8230
    Epoch 27/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4927 - accuracy: 0.8216
    Epoch 28/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4860 - accuracy: 0.8188
    Epoch 29/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4895 - accuracy: 0.8301
    Epoch 30/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4793 - accuracy: 0.8216
    Epoch 31/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4856 - accuracy: 0.8258
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4794 - accuracy: 0.8230
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4825 - accuracy: 0.8315
    Epoch 34/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4814 - accuracy: 0.8315
    Epoch 35/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4782 - accuracy: 0.8413
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4795 - accuracy: 0.8343
    Epoch 37/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4740 - accuracy: 0.8188
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4673 - accuracy: 0.8287
    Epoch 39/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4786 - accuracy: 0.8315
    Epoch 40/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4681 - accuracy: 0.8399
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4707 - accuracy: 0.8329
    Epoch 42/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4700 - accuracy: 0.8301
    Epoch 43/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4650 - accuracy: 0.8329
    Epoch 44/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4773 - accuracy: 0.8343
    Epoch 45/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4810 - accuracy: 0.8272
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4657 - accuracy: 0.8357
    Epoch 47/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4683 - accuracy: 0.8343
    Epoch 48/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4674 - accuracy: 0.8287
    Epoch 49/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4738 - accuracy: 0.8385
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4729 - accuracy: 0.8357
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.8746 - accuracy: 0.7195
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7550 - accuracy: 0.7770
    Epoch 3/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6868 - accuracy: 0.7882
    Epoch 4/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6522 - accuracy: 0.7896
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6075 - accuracy: 0.7924
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5858 - accuracy: 0.8121
    Epoch 7/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5722 - accuracy: 0.8149
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5683 - accuracy: 0.8233
    Epoch 9/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5460 - accuracy: 0.8149
    Epoch 10/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5423 - accuracy: 0.8191
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5344 - accuracy: 0.8079
    Epoch 12/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5250 - accuracy: 0.8191
    Epoch 13/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5113 - accuracy: 0.8205
    Epoch 14/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5136 - accuracy: 0.8121
    Epoch 15/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5105 - accuracy: 0.8121
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5026 - accuracy: 0.8149
    Epoch 17/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5016 - accuracy: 0.8233
    Epoch 18/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4965 - accuracy: 0.8289
    Epoch 19/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5036 - accuracy: 0.8177
    Epoch 20/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4924 - accuracy: 0.8233
    Epoch 21/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4785 - accuracy: 0.8247
    Epoch 22/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4986 - accuracy: 0.8149
    Epoch 23/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4770 - accuracy: 0.8359
    Epoch 24/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4722 - accuracy: 0.8331
    Epoch 25/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4891 - accuracy: 0.8289
    Epoch 26/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4889 - accuracy: 0.8177
    Epoch 27/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4796 - accuracy: 0.8303
    Epoch 28/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4756 - accuracy: 0.8317
    Epoch 29/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4807 - accuracy: 0.8303
    Epoch 30/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4798 - accuracy: 0.8303
    Epoch 31/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4738 - accuracy: 0.8331
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4751 - accuracy: 0.8345
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4764 - accuracy: 0.8345
    Epoch 34/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4737 - accuracy: 0.8261
    Epoch 35/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4721 - accuracy: 0.8205
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4761 - accuracy: 0.8331
    Epoch 37/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4689 - accuracy: 0.8247
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4735 - accuracy: 0.8261
    Epoch 39/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4725 - accuracy: 0.8289
    Epoch 40/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4694 - accuracy: 0.8289
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4658 - accuracy: 0.8317
    Epoch 42/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4573 - accuracy: 0.8387
    Epoch 43/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4655 - accuracy: 0.8429
    Epoch 44/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4758 - accuracy: 0.8205
    Epoch 45/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4670 - accuracy: 0.8331
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4703 - accuracy: 0.8443
    Epoch 47/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4654 - accuracy: 0.8373
    Epoch 48/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4560 - accuracy: 0.8303
    Epoch 49/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4758 - accuracy: 0.8205
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4603 - accuracy: 0.8233
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9503 - accuracy: 0.6410
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7997 - accuracy: 0.7447
    Epoch 3/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7338 - accuracy: 0.7700
    Epoch 4/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6832 - accuracy: 0.7952
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6439 - accuracy: 0.7980
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6342 - accuracy: 0.7994
    Epoch 7/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5970 - accuracy: 0.8093
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5812 - accuracy: 0.8247
    Epoch 9/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5706 - accuracy: 0.8177
    Epoch 10/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5476 - accuracy: 0.8205
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5506 - accuracy: 0.8135
    Epoch 12/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5440 - accuracy: 0.8219
    Epoch 13/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5242 - accuracy: 0.8401
    Epoch 14/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5285 - accuracy: 0.8135
    Epoch 15/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5230 - accuracy: 0.8275
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5066 - accuracy: 0.8331
    Epoch 17/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5181 - accuracy: 0.8289
    Epoch 18/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5185 - accuracy: 0.8191
    Epoch 19/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4996 - accuracy: 0.8303
    Epoch 20/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5017 - accuracy: 0.8303
    Epoch 21/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5034 - accuracy: 0.8261
    Epoch 22/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4999 - accuracy: 0.8247
    Epoch 23/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4878 - accuracy: 0.8443
    Epoch 24/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4945 - accuracy: 0.8261
    Epoch 25/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4966 - accuracy: 0.8261
    Epoch 26/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5030 - accuracy: 0.8303
    Epoch 27/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4975 - accuracy: 0.8331
    Epoch 28/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4908 - accuracy: 0.8345
    Epoch 29/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4916 - accuracy: 0.8345
    Epoch 30/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4953 - accuracy: 0.8261
    Epoch 31/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4797 - accuracy: 0.8415
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4888 - accuracy: 0.8401
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4842 - accuracy: 0.8233
    Epoch 34/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4840 - accuracy: 0.8345
    Epoch 35/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4787 - accuracy: 0.8359
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4795 - accuracy: 0.8345
    Epoch 37/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4755 - accuracy: 0.8331
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4727 - accuracy: 0.8289
    Epoch 39/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4820 - accuracy: 0.8261
    Epoch 40/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4756 - accuracy: 0.8331
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4749 - accuracy: 0.8331
    Epoch 42/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4693 - accuracy: 0.8331
    Epoch 43/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4726 - accuracy: 0.8429
    Epoch 44/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4700 - accuracy: 0.8345
    Epoch 45/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4721 - accuracy: 0.8415
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4734 - accuracy: 0.8415
    Epoch 47/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4672 - accuracy: 0.8485
    Epoch 48/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4703 - accuracy: 0.8401
    Epoch 49/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4748 - accuracy: 0.8387
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4751 - accuracy: 0.8415
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9268 - accuracy: 0.6367
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.8095 - accuracy: 0.6985
    Epoch 3/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7452 - accuracy: 0.7546
    Epoch 4/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7060 - accuracy: 0.7658
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6640 - accuracy: 0.8008
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6248 - accuracy: 0.8008
    Epoch 7/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6025 - accuracy: 0.8205
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6008 - accuracy: 0.8022
    Epoch 9/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5730 - accuracy: 0.8121
    Epoch 10/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5612 - accuracy: 0.8205
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5576 - accuracy: 0.8149
    Epoch 12/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5363 - accuracy: 0.8191
    Epoch 13/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5312 - accuracy: 0.8261
    Epoch 14/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5276 - accuracy: 0.8233
    Epoch 15/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5248 - accuracy: 0.8275
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5123 - accuracy: 0.8275
    Epoch 17/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5071 - accuracy: 0.8289
    Epoch 18/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5074 - accuracy: 0.8345
    Epoch 19/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5007 - accuracy: 0.8289
    Epoch 20/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4993 - accuracy: 0.8219
    Epoch 21/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4945 - accuracy: 0.8415
    Epoch 22/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4846 - accuracy: 0.8359
    Epoch 23/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4956 - accuracy: 0.8261
    Epoch 24/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4806 - accuracy: 0.8373
    Epoch 25/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4827 - accuracy: 0.8485
    Epoch 26/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4807 - accuracy: 0.8471
    Epoch 27/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4763 - accuracy: 0.8303
    Epoch 28/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4676 - accuracy: 0.8429
    Epoch 29/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4761 - accuracy: 0.8429
    Epoch 30/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4742 - accuracy: 0.8401
    Epoch 31/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4694 - accuracy: 0.8401
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4577 - accuracy: 0.8457
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4657 - accuracy: 0.8471
    Epoch 34/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4622 - accuracy: 0.8485
    Epoch 35/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4696 - accuracy: 0.8415
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4690 - accuracy: 0.8429
    Epoch 37/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4612 - accuracy: 0.8485
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4670 - accuracy: 0.8373
    Epoch 39/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4673 - accuracy: 0.8415
    Epoch 40/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4643 - accuracy: 0.8513
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4608 - accuracy: 0.8485
    Epoch 42/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4549 - accuracy: 0.8457
    Epoch 43/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4587 - accuracy: 0.8415
    Epoch 44/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4596 - accuracy: 0.8401
    Epoch 45/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4646 - accuracy: 0.8485
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4618 - accuracy: 0.8457
    Epoch 47/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4527 - accuracy: 0.8457
    Epoch 48/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4568 - accuracy: 0.8499
    Epoch 49/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4531 - accuracy: 0.8499
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4537 - accuracy: 0.8499
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9603 - accuracy: 0.5414
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.8541 - accuracy: 0.6788
    Epoch 3/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7813 - accuracy: 0.7475
    Epoch 4/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7310 - accuracy: 0.7546
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6970 - accuracy: 0.7518
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6497 - accuracy: 0.7686
    Epoch 7/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6243 - accuracy: 0.7784
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5975 - accuracy: 0.7826
    Epoch 9/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5914 - accuracy: 0.7714
    Epoch 10/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5716 - accuracy: 0.7896
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5497 - accuracy: 0.8022
    Epoch 12/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5428 - accuracy: 0.7994
    Epoch 13/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5294 - accuracy: 0.8079
    Epoch 14/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5344 - accuracy: 0.8022
    Epoch 15/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5209 - accuracy: 0.8107
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5263 - accuracy: 0.8121
    Epoch 17/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5220 - accuracy: 0.8036
    Epoch 18/50
    45/45 [==============================] - 0s 6ms/step - loss: 0.5197 - accuracy: 0.8065
    Epoch 19/50
    45/45 [==============================] - 0s 6ms/step - loss: 0.5144 - accuracy: 0.8219
    Epoch 20/50
    45/45 [==============================] - 0s 5ms/step - loss: 0.5136 - accuracy: 0.8079
    Epoch 21/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5119 - accuracy: 0.8149
    Epoch 22/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5110 - accuracy: 0.8135
    Epoch 23/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4992 - accuracy: 0.8163
    Epoch 24/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5034 - accuracy: 0.8022
    Epoch 25/50
    45/45 [==============================] - 0s 8ms/step - loss: 0.5090 - accuracy: 0.7994
    Epoch 26/50
    45/45 [==============================] - 0s 6ms/step - loss: 0.4998 - accuracy: 0.8135
    Epoch 27/50
    45/45 [==============================] - 0s 4ms/step - loss: 0.4951 - accuracy: 0.8177
    Epoch 28/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5060 - accuracy: 0.8079
    Epoch 29/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5021 - accuracy: 0.8163
    Epoch 30/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4951 - accuracy: 0.8135
    Epoch 31/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4946 - accuracy: 0.8247
    Epoch 32/50
    45/45 [==============================] - 0s 4ms/step - loss: 0.4963 - accuracy: 0.8177
    Epoch 33/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4875 - accuracy: 0.8205
    Epoch 34/50
    45/45 [==============================] - 0s 7ms/step - loss: 0.4915 - accuracy: 0.8247
    Epoch 35/50
    45/45 [==============================] - 0s 6ms/step - loss: 0.4905 - accuracy: 0.8205
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4875 - accuracy: 0.8149
    Epoch 37/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4869 - accuracy: 0.8121
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4886 - accuracy: 0.8079
    Epoch 39/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4881 - accuracy: 0.8163
    Epoch 40/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4906 - accuracy: 0.8233
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4857 - accuracy: 0.8205
    Epoch 42/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4981 - accuracy: 0.8149
    Epoch 43/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4869 - accuracy: 0.8079
    Epoch 44/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4905 - accuracy: 0.8149
    Epoch 45/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4836 - accuracy: 0.8233
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4861 - accuracy: 0.8205
    Epoch 47/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4972 - accuracy: 0.8233
    Epoch 48/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4824 - accuracy: 0.8177
    Epoch 49/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4787 - accuracy: 0.8401
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4800 - accuracy: 0.8261
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 2ms/step - loss: 0.9643 - accuracy: 0.6447
    Epoch 2/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.8678 - accuracy: 0.7149
    Epoch 3/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7765 - accuracy: 0.7725
    Epoch 4/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7123 - accuracy: 0.8006
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6770 - accuracy: 0.7963
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6470 - accuracy: 0.7907
    Epoch 7/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6213 - accuracy: 0.7935
    Epoch 8/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6048 - accuracy: 0.7949
    Epoch 9/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5837 - accuracy: 0.8062
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5753 - accuracy: 0.8174
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 2ms/step - loss: 0.9570 - accuracy: 0.6522
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8373 - accuracy: 0.7924
    Epoch 3/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7506 - accuracy: 0.7980
    Epoch 4/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6963 - accuracy: 0.8079
    Epoch 5/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6573 - accuracy: 0.8261
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6395 - accuracy: 0.8121
    Epoch 7/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6132 - accuracy: 0.8303
    Epoch 8/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6033 - accuracy: 0.8205
    Epoch 9/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5908 - accuracy: 0.8247
    Epoch 10/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5760 - accuracy: 0.8205
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 2ms/step - loss: 0.9246 - accuracy: 0.6620
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8287 - accuracy: 0.6900
    Epoch 3/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7672 - accuracy: 0.7307
    Epoch 4/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7244 - accuracy: 0.7546
    Epoch 5/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6944 - accuracy: 0.7728
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6550 - accuracy: 0.7910
    Epoch 7/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6403 - accuracy: 0.8050
    Epoch 8/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6082 - accuracy: 0.8093
    Epoch 9/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5859 - accuracy: 0.8107
    Epoch 10/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5780 - accuracy: 0.8079
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 2ms/step - loss: 0.9784 - accuracy: 0.6381
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8675 - accuracy: 0.6858
    Epoch 3/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7952 - accuracy: 0.7321
    Epoch 4/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7383 - accuracy: 0.7616
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6962 - accuracy: 0.7826
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6718 - accuracy: 0.7896
    Epoch 7/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6378 - accuracy: 0.8177
    Epoch 8/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6221 - accuracy: 0.8121
    Epoch 9/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5968 - accuracy: 0.8121
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5836 - accuracy: 0.8219
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 2ms/step - loss: 0.9058 - accuracy: 0.6802
    Epoch 2/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7914 - accuracy: 0.7658
    Epoch 3/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.7258 - accuracy: 0.7938
    Epoch 4/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6913 - accuracy: 0.8022
    Epoch 5/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6617 - accuracy: 0.7980
    Epoch 6/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6524 - accuracy: 0.7980
    Epoch 7/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6211 - accuracy: 0.7854
    Epoch 8/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.6030 - accuracy: 0.8008
    Epoch 9/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5818 - accuracy: 0.8121
    Epoch 10/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.5811 - accuracy: 0.8036
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 2ms/step - loss: 0.9424 - accuracy: 0.6896
    Epoch 2/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.8293 - accuracy: 0.7458
    Epoch 3/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7756 - accuracy: 0.7289
    Epoch 4/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7008 - accuracy: 0.7640
    Epoch 5/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6668 - accuracy: 0.7753
    Epoch 6/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6487 - accuracy: 0.7823
    Epoch 7/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6274 - accuracy: 0.7781
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6171 - accuracy: 0.8020
    Epoch 9/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5937 - accuracy: 0.7865
    Epoch 10/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5843 - accuracy: 0.8034
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5648 - accuracy: 0.8062
    Epoch 12/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5645 - accuracy: 0.7949
    Epoch 13/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5508 - accuracy: 0.8062
    Epoch 14/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5436 - accuracy: 0.8076
    Epoch 15/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5386 - accuracy: 0.8118
    Epoch 16/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5258 - accuracy: 0.8160
    Epoch 17/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5150 - accuracy: 0.8174
    Epoch 18/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5200 - accuracy: 0.8020
    Epoch 19/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5145 - accuracy: 0.8160
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5063 - accuracy: 0.8216
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 2ms/step - loss: 0.9310 - accuracy: 0.6424
    Epoch 2/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8314 - accuracy: 0.7097
    Epoch 3/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7742 - accuracy: 0.7433
    Epoch 4/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7258 - accuracy: 0.7728
    Epoch 5/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7060 - accuracy: 0.7728
    Epoch 6/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6704 - accuracy: 0.8036
    Epoch 7/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6451 - accuracy: 0.7924
    Epoch 8/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6145 - accuracy: 0.7966
    Epoch 9/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5999 - accuracy: 0.8022
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5965 - accuracy: 0.8093
    Epoch 11/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5737 - accuracy: 0.8121
    Epoch 12/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5617 - accuracy: 0.8163
    Epoch 13/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5678 - accuracy: 0.8121
    Epoch 14/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5502 - accuracy: 0.8149
    Epoch 15/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5363 - accuracy: 0.8261
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5390 - accuracy: 0.8233
    Epoch 17/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5240 - accuracy: 0.8163
    Epoch 18/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5176 - accuracy: 0.8219
    Epoch 19/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5134 - accuracy: 0.8205
    Epoch 20/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5167 - accuracy: 0.8247
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 2ms/step - loss: 0.9455 - accuracy: 0.6522
    Epoch 2/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.8252 - accuracy: 0.7293
    Epoch 3/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7519 - accuracy: 0.7532
    Epoch 4/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6906 - accuracy: 0.7742
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6533 - accuracy: 0.7812
    Epoch 6/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6409 - accuracy: 0.7882
    Epoch 7/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6092 - accuracy: 0.7952
    Epoch 8/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5866 - accuracy: 0.7994
    Epoch 9/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5795 - accuracy: 0.8107
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5580 - accuracy: 0.8177
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5554 - accuracy: 0.8163
    Epoch 12/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5622 - accuracy: 0.8121
    Epoch 13/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5550 - accuracy: 0.8036
    Epoch 14/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5291 - accuracy: 0.8107
    Epoch 15/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5189 - accuracy: 0.8233
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5300 - accuracy: 0.8191
    Epoch 17/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5136 - accuracy: 0.8247
    Epoch 18/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5247 - accuracy: 0.8135
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5175 - accuracy: 0.8205
    Epoch 20/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5139 - accuracy: 0.8177
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 2ms/step - loss: 0.9665 - accuracy: 0.6353
    Epoch 2/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.8367 - accuracy: 0.7546
    Epoch 3/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7559 - accuracy: 0.7742
    Epoch 4/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.7049 - accuracy: 0.7728
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6800 - accuracy: 0.7854
    Epoch 6/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6348 - accuracy: 0.7980
    Epoch 7/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6158 - accuracy: 0.8036
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6078 - accuracy: 0.8036
    Epoch 9/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5884 - accuracy: 0.7924
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5811 - accuracy: 0.8149
    Epoch 11/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5791 - accuracy: 0.8079
    Epoch 12/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5576 - accuracy: 0.8065
    Epoch 13/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5613 - accuracy: 0.8177
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5488 - accuracy: 0.8261
    Epoch 15/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5331 - accuracy: 0.8205
    Epoch 16/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5225 - accuracy: 0.8247
    Epoch 17/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5254 - accuracy: 0.8359
    Epoch 18/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5129 - accuracy: 0.8233
    Epoch 19/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5033 - accuracy: 0.8345
    Epoch 20/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5018 - accuracy: 0.8373
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 3ms/step - loss: 0.9464 - accuracy: 0.6452
    Epoch 2/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.8370 - accuracy: 0.7546
    Epoch 3/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7582 - accuracy: 0.7826
    Epoch 4/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6944 - accuracy: 0.8050
    Epoch 5/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6686 - accuracy: 0.7868
    Epoch 6/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6467 - accuracy: 0.8036
    Epoch 7/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.6236 - accuracy: 0.8008
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5905 - accuracy: 0.8163
    Epoch 9/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5795 - accuracy: 0.8107
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5693 - accuracy: 0.8093
    Epoch 11/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5630 - accuracy: 0.8036
    Epoch 12/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5514 - accuracy: 0.8177
    Epoch 13/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5581 - accuracy: 0.8079
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5374 - accuracy: 0.8177
    Epoch 15/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5495 - accuracy: 0.8219
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5445 - accuracy: 0.8065
    Epoch 17/20
    29/29 [==============================] - 0s 2ms/step - loss: 0.5319 - accuracy: 0.8065
    Epoch 18/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5412 - accuracy: 0.7938
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5297 - accuracy: 0.8050
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5267 - accuracy: 0.8079
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 2ms/step - loss: 0.9463 - accuracy: 0.6840
    Epoch 2/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8107 - accuracy: 0.7893
    Epoch 3/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.7390 - accuracy: 0.7935
    Epoch 4/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6949 - accuracy: 0.7907
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6627 - accuracy: 0.8076
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6432 - accuracy: 0.8104
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6182 - accuracy: 0.8118
    Epoch 8/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5999 - accuracy: 0.8146
    Epoch 9/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5786 - accuracy: 0.8216
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5781 - accuracy: 0.8216
    Epoch 11/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5578 - accuracy: 0.8174
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5649 - accuracy: 0.8174
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5434 - accuracy: 0.8329
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5335 - accuracy: 0.8329
    Epoch 15/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5300 - accuracy: 0.8188
    Epoch 16/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5162 - accuracy: 0.8202
    Epoch 17/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5254 - accuracy: 0.8188
    Epoch 18/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5255 - accuracy: 0.8244
    Epoch 19/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5073 - accuracy: 0.8216
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5061 - accuracy: 0.8357
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5064 - accuracy: 0.8287
    Epoch 22/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5003 - accuracy: 0.8216
    Epoch 23/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4967 - accuracy: 0.8371
    Epoch 24/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5008 - accuracy: 0.8244
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4920 - accuracy: 0.8258
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4874 - accuracy: 0.8343
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4833 - accuracy: 0.8315
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4780 - accuracy: 0.8385
    Epoch 29/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4853 - accuracy: 0.8315
    Epoch 30/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4788 - accuracy: 0.8329
    Epoch 31/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4816 - accuracy: 0.8315
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4805 - accuracy: 0.8315
    Epoch 33/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4833 - accuracy: 0.8301
    Epoch 34/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4790 - accuracy: 0.8343
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4738 - accuracy: 0.8287
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4755 - accuracy: 0.8343
    Epoch 37/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4783 - accuracy: 0.8272
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4813 - accuracy: 0.8230
    Epoch 39/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4730 - accuracy: 0.8258
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4736 - accuracy: 0.8244
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4635 - accuracy: 0.8371
    Epoch 42/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4648 - accuracy: 0.8343
    Epoch 43/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4730 - accuracy: 0.8371
    Epoch 44/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4808 - accuracy: 0.8272
    Epoch 45/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4651 - accuracy: 0.8343
    Epoch 46/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4682 - accuracy: 0.8455
    Epoch 47/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4726 - accuracy: 0.8287
    Epoch 48/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4798 - accuracy: 0.8272
    Epoch 49/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4761 - accuracy: 0.8329
    Epoch 50/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4652 - accuracy: 0.8287
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 2ms/step - loss: 0.9661 - accuracy: 0.5708
    Epoch 2/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.8453 - accuracy: 0.7013
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7699 - accuracy: 0.7644
    Epoch 4/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7273 - accuracy: 0.7616
    Epoch 5/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6676 - accuracy: 0.7994
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6584 - accuracy: 0.8008
    Epoch 7/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6296 - accuracy: 0.7966
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6195 - accuracy: 0.7910
    Epoch 9/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6125 - accuracy: 0.7966
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5948 - accuracy: 0.8022
    Epoch 11/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5788 - accuracy: 0.8121
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5699 - accuracy: 0.8079
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5534 - accuracy: 0.8121
    Epoch 14/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5508 - accuracy: 0.8121
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5304 - accuracy: 0.8275
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5342 - accuracy: 0.8219
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5179 - accuracy: 0.8331
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5286 - accuracy: 0.8275
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5233 - accuracy: 0.8205
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5194 - accuracy: 0.8191
    Epoch 21/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5178 - accuracy: 0.8149
    Epoch 22/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5060 - accuracy: 0.8303
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5138 - accuracy: 0.8219
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5086 - accuracy: 0.8303
    Epoch 25/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5039 - accuracy: 0.8233
    Epoch 26/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5053 - accuracy: 0.8261
    Epoch 27/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4979 - accuracy: 0.8205
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5047 - accuracy: 0.8261
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4918 - accuracy: 0.8247
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4868 - accuracy: 0.8373
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4891 - accuracy: 0.8275
    Epoch 32/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4863 - accuracy: 0.8261
    Epoch 33/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4878 - accuracy: 0.8345
    Epoch 34/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4895 - accuracy: 0.8373
    Epoch 35/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4870 - accuracy: 0.8303
    Epoch 36/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4958 - accuracy: 0.8191
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4759 - accuracy: 0.8359
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4797 - accuracy: 0.8261
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4818 - accuracy: 0.8289
    Epoch 40/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4690 - accuracy: 0.8331
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4795 - accuracy: 0.8177
    Epoch 42/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4683 - accuracy: 0.8415
    Epoch 43/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4777 - accuracy: 0.8303
    Epoch 44/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4747 - accuracy: 0.8261
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4757 - accuracy: 0.8177
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4701 - accuracy: 0.8261
    Epoch 47/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4677 - accuracy: 0.8205
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4621 - accuracy: 0.8415
    Epoch 49/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4654 - accuracy: 0.8345
    Epoch 50/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4658 - accuracy: 0.8331
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.9619 - accuracy: 0.6283
    Epoch 2/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.8542 - accuracy: 0.7237
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7932 - accuracy: 0.7700
    Epoch 4/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.7450 - accuracy: 0.7798
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7204 - accuracy: 0.7630
    Epoch 6/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6731 - accuracy: 0.8065
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6624 - accuracy: 0.7840
    Epoch 8/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6483 - accuracy: 0.7784
    Epoch 9/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6224 - accuracy: 0.7882
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5942 - accuracy: 0.8219
    Epoch 11/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5923 - accuracy: 0.8093
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5567 - accuracy: 0.8079
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5475 - accuracy: 0.8303
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5538 - accuracy: 0.8163
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5456 - accuracy: 0.8233
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5430 - accuracy: 0.8149
    Epoch 17/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5315 - accuracy: 0.8177
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5219 - accuracy: 0.8317
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5281 - accuracy: 0.8247
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5128 - accuracy: 0.8261
    Epoch 21/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5206 - accuracy: 0.8261
    Epoch 22/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5064 - accuracy: 0.8303
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5056 - accuracy: 0.8219
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5142 - accuracy: 0.8261
    Epoch 25/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4995 - accuracy: 0.8331
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5035 - accuracy: 0.8345
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5084 - accuracy: 0.8317
    Epoch 28/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4923 - accuracy: 0.8373
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4963 - accuracy: 0.8415
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4973 - accuracy: 0.8275
    Epoch 31/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5034 - accuracy: 0.8359
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4927 - accuracy: 0.8303
    Epoch 33/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5080 - accuracy: 0.8163
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4940 - accuracy: 0.8317
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4881 - accuracy: 0.8303
    Epoch 36/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4970 - accuracy: 0.8261
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4967 - accuracy: 0.8275
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4857 - accuracy: 0.8401
    Epoch 39/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4987 - accuracy: 0.8247
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4930 - accuracy: 0.8387
    Epoch 41/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4876 - accuracy: 0.8303
    Epoch 42/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4828 - accuracy: 0.8415
    Epoch 43/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4857 - accuracy: 0.8345
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4718 - accuracy: 0.8359
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4833 - accuracy: 0.8317
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4848 - accuracy: 0.8387
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4837 - accuracy: 0.8289
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4921 - accuracy: 0.8359
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4893 - accuracy: 0.8261
    Epoch 50/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4898 - accuracy: 0.8261
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 2ms/step - loss: 0.9369 - accuracy: 0.6746
    Epoch 2/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8363 - accuracy: 0.7405
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7676 - accuracy: 0.7742
    Epoch 4/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.7103 - accuracy: 0.8008
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6727 - accuracy: 0.8022
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6624 - accuracy: 0.8022
    Epoch 7/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6235 - accuracy: 0.8149
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6075 - accuracy: 0.8233
    Epoch 9/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5974 - accuracy: 0.8135
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5720 - accuracy: 0.8261
    Epoch 11/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5691 - accuracy: 0.8205
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5617 - accuracy: 0.8163
    Epoch 13/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5510 - accuracy: 0.8149
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5475 - accuracy: 0.8233
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5350 - accuracy: 0.8275
    Epoch 16/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5348 - accuracy: 0.8317
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5249 - accuracy: 0.8345
    Epoch 18/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5189 - accuracy: 0.8331
    Epoch 19/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5124 - accuracy: 0.8261
    Epoch 20/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5228 - accuracy: 0.8345
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5165 - accuracy: 0.8289
    Epoch 22/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5110 - accuracy: 0.8317
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4949 - accuracy: 0.8387
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4994 - accuracy: 0.8387
    Epoch 25/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5013 - accuracy: 0.8359
    Epoch 26/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5030 - accuracy: 0.8401
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5003 - accuracy: 0.8373
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4926 - accuracy: 0.8359
    Epoch 29/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4931 - accuracy: 0.8457
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4842 - accuracy: 0.8317
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4889 - accuracy: 0.8401
    Epoch 32/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4863 - accuracy: 0.8359
    Epoch 33/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4848 - accuracy: 0.8401
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4791 - accuracy: 0.8429
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4839 - accuracy: 0.8457
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4752 - accuracy: 0.8429
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4709 - accuracy: 0.8429
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4729 - accuracy: 0.8471
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4890 - accuracy: 0.8345
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4760 - accuracy: 0.8443
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4710 - accuracy: 0.8471
    Epoch 42/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4655 - accuracy: 0.8429
    Epoch 43/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4678 - accuracy: 0.8499
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4667 - accuracy: 0.8387
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4824 - accuracy: 0.8429
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4779 - accuracy: 0.8373
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4712 - accuracy: 0.8471
    Epoch 48/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4607 - accuracy: 0.8401
    Epoch 49/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4728 - accuracy: 0.8345
    Epoch 50/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4601 - accuracy: 0.8513
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.9552 - accuracy: 0.6045
    Epoch 2/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8575 - accuracy: 0.7027
    Epoch 3/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.7870 - accuracy: 0.7532
    Epoch 4/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7422 - accuracy: 0.7644
    Epoch 5/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.7088 - accuracy: 0.7756
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6769 - accuracy: 0.7784
    Epoch 7/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.6491 - accuracy: 0.7854
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6315 - accuracy: 0.8022
    Epoch 9/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6166 - accuracy: 0.8093
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5978 - accuracy: 0.7952
    Epoch 11/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5791 - accuracy: 0.8093
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5674 - accuracy: 0.8093
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5684 - accuracy: 0.8093
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5590 - accuracy: 0.8107
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5379 - accuracy: 0.8079
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5431 - accuracy: 0.8121
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5430 - accuracy: 0.8093
    Epoch 18/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5333 - accuracy: 0.8036
    Epoch 19/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5189 - accuracy: 0.8191
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5276 - accuracy: 0.8205
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5318 - accuracy: 0.8149
    Epoch 22/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5181 - accuracy: 0.8107
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5129 - accuracy: 0.8121
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5214 - accuracy: 0.8135
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4963 - accuracy: 0.8219
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5073 - accuracy: 0.8149
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5148 - accuracy: 0.8093
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5022 - accuracy: 0.8163
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5000 - accuracy: 0.8219
    Epoch 30/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5145 - accuracy: 0.8135
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5024 - accuracy: 0.8191
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5032 - accuracy: 0.8149
    Epoch 33/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.5032 - accuracy: 0.8191
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4991 - accuracy: 0.8177
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5016 - accuracy: 0.8107
    Epoch 36/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4909 - accuracy: 0.8247
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5059 - accuracy: 0.8177
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4963 - accuracy: 0.8177
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4881 - accuracy: 0.8205
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4980 - accuracy: 0.8233
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4868 - accuracy: 0.8205
    Epoch 42/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4952 - accuracy: 0.8247
    Epoch 43/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4943 - accuracy: 0.8261
    Epoch 44/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4906 - accuracy: 0.8149
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4980 - accuracy: 0.8135
    Epoch 46/50
    29/29 [==============================] - 0s 2ms/step - loss: 0.4854 - accuracy: 0.8247
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4956 - accuracy: 0.8149
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4908 - accuracy: 0.8247
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4936 - accuracy: 0.8177
    Epoch 50/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4863 - accuracy: 0.8205
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 2ms/step - loss: 0.9727 - accuracy: 0.5787
    Epoch 2/10
    23/23 [==============================] - 0s 2ms/step - loss: 0.8887 - accuracy: 0.7107
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8126 - accuracy: 0.7360
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7546 - accuracy: 0.7486
    Epoch 5/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7115 - accuracy: 0.7697
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6676 - accuracy: 0.7893
    Epoch 7/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6442 - accuracy: 0.7963
    Epoch 8/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6351 - accuracy: 0.7739
    Epoch 9/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6077 - accuracy: 0.8062
    Epoch 10/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.5956 - accuracy: 0.8006
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 2ms/step - loss: 0.9607 - accuracy: 0.6872
    Epoch 2/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8474 - accuracy: 0.7616
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7793 - accuracy: 0.7742
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7251 - accuracy: 0.7756
    Epoch 5/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6774 - accuracy: 0.7896
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6546 - accuracy: 0.7868
    Epoch 7/10
    23/23 [==============================] - 0s 2ms/step - loss: 0.6381 - accuracy: 0.8008
    Epoch 8/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6192 - accuracy: 0.7966
    Epoch 9/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.5977 - accuracy: 0.8008
    Epoch 10/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.5932 - accuracy: 0.8149
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 3ms/step - loss: 0.9550 - accuracy: 0.6550
    Epoch 2/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8502 - accuracy: 0.7504
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7852 - accuracy: 0.7742
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7368 - accuracy: 0.7812
    Epoch 5/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7386 - accuracy: 0.7728
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6920 - accuracy: 0.7910
    Epoch 7/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6785 - accuracy: 0.7854
    Epoch 8/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6609 - accuracy: 0.7924
    Epoch 9/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6517 - accuracy: 0.7812
    Epoch 10/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6377 - accuracy: 0.7826
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 3ms/step - loss: 0.9482 - accuracy: 0.6410
    Epoch 2/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8102 - accuracy: 0.7994
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7430 - accuracy: 0.7966
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6968 - accuracy: 0.8079
    Epoch 5/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6745 - accuracy: 0.8065
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6458 - accuracy: 0.8163
    Epoch 7/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6367 - accuracy: 0.8008
    Epoch 8/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6047 - accuracy: 0.8093
    Epoch 9/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6066 - accuracy: 0.8093
    Epoch 10/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.5824 - accuracy: 0.8022
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 2ms/step - loss: 0.9669 - accuracy: 0.6045
    Epoch 2/10
    23/23 [==============================] - 0s 2ms/step - loss: 0.8793 - accuracy: 0.6914
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8195 - accuracy: 0.7111
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7642 - accuracy: 0.7518
    Epoch 5/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7278 - accuracy: 0.7672
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6951 - accuracy: 0.7700
    Epoch 7/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6701 - accuracy: 0.7868
    Epoch 8/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6438 - accuracy: 0.7924
    Epoch 9/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6345 - accuracy: 0.7868
    Epoch 10/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6247 - accuracy: 0.7812
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 2ms/step - loss: 0.9416 - accuracy: 0.7233
    Epoch 2/20
    23/23 [==============================] - 0s 2ms/step - loss: 0.8635 - accuracy: 0.7317
    Epoch 3/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7977 - accuracy: 0.7669
    Epoch 4/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7324 - accuracy: 0.7683
    Epoch 5/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6938 - accuracy: 0.7907
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6737 - accuracy: 0.7949
    Epoch 7/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6570 - accuracy: 0.7935
    Epoch 8/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6317 - accuracy: 0.7893
    Epoch 9/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6310 - accuracy: 0.7837
    Epoch 10/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5979 - accuracy: 0.8104
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6130 - accuracy: 0.7837
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5933 - accuracy: 0.8062
    Epoch 13/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5770 - accuracy: 0.7992
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5872 - accuracy: 0.7992
    Epoch 15/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5625 - accuracy: 0.8062
    Epoch 16/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5701 - accuracy: 0.7978
    Epoch 17/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5490 - accuracy: 0.8132
    Epoch 18/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5259 - accuracy: 0.8244
    Epoch 19/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5413 - accuracy: 0.8090
    Epoch 20/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5332 - accuracy: 0.8076
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 0.9523 - accuracy: 0.7251
    Epoch 2/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.8745 - accuracy: 0.7588
    Epoch 3/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7949 - accuracy: 0.7770
    Epoch 4/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7382 - accuracy: 0.7966
    Epoch 5/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7086 - accuracy: 0.7910
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6824 - accuracy: 0.7784
    Epoch 7/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6667 - accuracy: 0.7896
    Epoch 8/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6449 - accuracy: 0.7840
    Epoch 9/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6170 - accuracy: 0.8008
    Epoch 10/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6167 - accuracy: 0.7882
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6048 - accuracy: 0.7882
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6015 - accuracy: 0.7924
    Epoch 13/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5814 - accuracy: 0.8163
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5669 - accuracy: 0.7994
    Epoch 15/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5592 - accuracy: 0.8022
    Epoch 16/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5490 - accuracy: 0.8219
    Epoch 17/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5490 - accuracy: 0.8079
    Epoch 18/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5444 - accuracy: 0.8093
    Epoch 19/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5410 - accuracy: 0.8177
    Epoch 20/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5243 - accuracy: 0.8191
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 1.0004 - accuracy: 0.6241
    Epoch 2/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.9079 - accuracy: 0.6676
    Epoch 3/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.8474 - accuracy: 0.7125
    Epoch 4/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7962 - accuracy: 0.7237
    Epoch 5/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7633 - accuracy: 0.7433
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7190 - accuracy: 0.7700
    Epoch 7/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6949 - accuracy: 0.7854
    Epoch 8/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6615 - accuracy: 0.7882
    Epoch 9/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6435 - accuracy: 0.8093
    Epoch 10/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6283 - accuracy: 0.8079
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6154 - accuracy: 0.8008
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5969 - accuracy: 0.8135
    Epoch 13/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5892 - accuracy: 0.8177
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5805 - accuracy: 0.8205
    Epoch 15/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5659 - accuracy: 0.8191
    Epoch 16/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5657 - accuracy: 0.8205
    Epoch 17/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5525 - accuracy: 0.8205
    Epoch 18/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5546 - accuracy: 0.8247
    Epoch 19/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5348 - accuracy: 0.8205
    Epoch 20/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5404 - accuracy: 0.8247
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 0.9857 - accuracy: 0.6129
    Epoch 2/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.8854 - accuracy: 0.7546
    Epoch 3/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.8065 - accuracy: 0.7574
    Epoch 4/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7495 - accuracy: 0.7728
    Epoch 5/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7222 - accuracy: 0.7756
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6763 - accuracy: 0.8050
    Epoch 7/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6682 - accuracy: 0.7938
    Epoch 8/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6555 - accuracy: 0.7980
    Epoch 9/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6359 - accuracy: 0.8050
    Epoch 10/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6183 - accuracy: 0.8079
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5927 - accuracy: 0.8275
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5851 - accuracy: 0.8079
    Epoch 13/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5772 - accuracy: 0.8107
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5724 - accuracy: 0.8149
    Epoch 15/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5598 - accuracy: 0.8289
    Epoch 16/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5533 - accuracy: 0.8191
    Epoch 17/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5509 - accuracy: 0.8121
    Epoch 18/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5417 - accuracy: 0.8121
    Epoch 19/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5230 - accuracy: 0.8233
    Epoch 20/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5244 - accuracy: 0.8233
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 2ms/step - loss: 0.9800 - accuracy: 0.5835
    Epoch 2/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.8840 - accuracy: 0.6928
    Epoch 3/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.8213 - accuracy: 0.7125
    Epoch 4/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7654 - accuracy: 0.7391
    Epoch 5/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7302 - accuracy: 0.7335
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6983 - accuracy: 0.7812
    Epoch 7/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6604 - accuracy: 0.7840
    Epoch 8/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6366 - accuracy: 0.7924
    Epoch 9/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6192 - accuracy: 0.7994
    Epoch 10/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6026 - accuracy: 0.7924
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5887 - accuracy: 0.8079
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5833 - accuracy: 0.7938
    Epoch 13/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5736 - accuracy: 0.8093
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5681 - accuracy: 0.8093
    Epoch 15/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5617 - accuracy: 0.8036
    Epoch 16/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5574 - accuracy: 0.8107
    Epoch 17/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5482 - accuracy: 0.8149
    Epoch 18/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5405 - accuracy: 0.8163
    Epoch 19/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5400 - accuracy: 0.8065
    Epoch 20/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5438 - accuracy: 0.8079
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 2ms/step - loss: 0.9986 - accuracy: 0.5829
    Epoch 2/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8973 - accuracy: 0.7177
    Epoch 3/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8233 - accuracy: 0.7725
    Epoch 4/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7668 - accuracy: 0.7640
    Epoch 5/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7480 - accuracy: 0.7654
    Epoch 6/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6986 - accuracy: 0.7837
    Epoch 7/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6877 - accuracy: 0.7767
    Epoch 8/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6608 - accuracy: 0.7935
    Epoch 9/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6421 - accuracy: 0.7907
    Epoch 10/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6174 - accuracy: 0.8020
    Epoch 11/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6058 - accuracy: 0.7865
    Epoch 12/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5905 - accuracy: 0.7907
    Epoch 13/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5774 - accuracy: 0.7963
    Epoch 14/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5688 - accuracy: 0.8132
    Epoch 15/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5617 - accuracy: 0.8034
    Epoch 16/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5574 - accuracy: 0.7978
    Epoch 17/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5415 - accuracy: 0.8090
    Epoch 18/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5306 - accuracy: 0.8104
    Epoch 19/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5340 - accuracy: 0.8048
    Epoch 20/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5276 - accuracy: 0.8132
    Epoch 21/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5257 - accuracy: 0.8160
    Epoch 22/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5076 - accuracy: 0.8357
    Epoch 23/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5141 - accuracy: 0.8258
    Epoch 24/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5066 - accuracy: 0.8287
    Epoch 25/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5067 - accuracy: 0.8258
    Epoch 26/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5046 - accuracy: 0.8230
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4851 - accuracy: 0.8315
    Epoch 28/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5012 - accuracy: 0.8244
    Epoch 29/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4875 - accuracy: 0.8315
    Epoch 30/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4845 - accuracy: 0.8160
    Epoch 31/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4851 - accuracy: 0.8301
    Epoch 32/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4889 - accuracy: 0.8244
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4865 - accuracy: 0.8385
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4811 - accuracy: 0.8343
    Epoch 35/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4837 - accuracy: 0.8357
    Epoch 36/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4779 - accuracy: 0.8244
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4822 - accuracy: 0.8244
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4676 - accuracy: 0.8315
    Epoch 39/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4720 - accuracy: 0.8315
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4696 - accuracy: 0.8329
    Epoch 41/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4687 - accuracy: 0.8385
    Epoch 42/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4749 - accuracy: 0.8329
    Epoch 43/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4722 - accuracy: 0.8272
    Epoch 44/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4681 - accuracy: 0.8160
    Epoch 45/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4674 - accuracy: 0.8441
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4632 - accuracy: 0.8427
    Epoch 47/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4696 - accuracy: 0.8413
    Epoch 48/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4590 - accuracy: 0.8399
    Epoch 49/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4541 - accuracy: 0.8441
    Epoch 50/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4760 - accuracy: 0.8301
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 3ms/step - loss: 1.0023 - accuracy: 0.5484
    Epoch 2/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8849 - accuracy: 0.6732
    Epoch 3/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8039 - accuracy: 0.7560
    Epoch 4/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7617 - accuracy: 0.7812
    Epoch 5/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7107 - accuracy: 0.7980
    Epoch 6/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6787 - accuracy: 0.7952
    Epoch 7/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6554 - accuracy: 0.8107
    Epoch 8/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6343 - accuracy: 0.8163
    Epoch 9/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6251 - accuracy: 0.8093
    Epoch 10/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6166 - accuracy: 0.7966
    Epoch 11/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5974 - accuracy: 0.8177
    Epoch 12/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5871 - accuracy: 0.8036
    Epoch 13/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5850 - accuracy: 0.7966
    Epoch 14/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5718 - accuracy: 0.8107
    Epoch 15/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5577 - accuracy: 0.8149
    Epoch 16/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5255 - accuracy: 0.8345
    Epoch 17/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5393 - accuracy: 0.8219
    Epoch 18/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5485 - accuracy: 0.8121
    Epoch 19/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5331 - accuracy: 0.8149
    Epoch 20/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5328 - accuracy: 0.8275
    Epoch 21/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5252 - accuracy: 0.8303
    Epoch 22/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5209 - accuracy: 0.8289
    Epoch 23/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5125 - accuracy: 0.8275
    Epoch 24/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5089 - accuracy: 0.8219
    Epoch 25/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5182 - accuracy: 0.8149
    Epoch 26/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4982 - accuracy: 0.8331
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5111 - accuracy: 0.8233
    Epoch 28/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4873 - accuracy: 0.8317
    Epoch 29/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4915 - accuracy: 0.8387
    Epoch 30/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4929 - accuracy: 0.8387
    Epoch 31/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4993 - accuracy: 0.8289
    Epoch 32/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4984 - accuracy: 0.8373
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4892 - accuracy: 0.8233
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4844 - accuracy: 0.8331
    Epoch 35/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4858 - accuracy: 0.8289
    Epoch 36/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4879 - accuracy: 0.8233
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4878 - accuracy: 0.8219
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4828 - accuracy: 0.8373
    Epoch 39/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4807 - accuracy: 0.8289
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4832 - accuracy: 0.8247
    Epoch 41/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4848 - accuracy: 0.8261
    Epoch 42/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4765 - accuracy: 0.8401
    Epoch 43/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4814 - accuracy: 0.8387
    Epoch 44/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4810 - accuracy: 0.8387
    Epoch 45/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4706 - accuracy: 0.8317
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4768 - accuracy: 0.8345
    Epoch 47/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4680 - accuracy: 0.8331
    Epoch 48/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4803 - accuracy: 0.8359
    Epoch 49/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4692 - accuracy: 0.8261
    Epoch 50/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4823 - accuracy: 0.8247
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 3ms/step - loss: 0.9376 - accuracy: 0.6985
    Epoch 2/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8439 - accuracy: 0.7588
    Epoch 3/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7945 - accuracy: 0.7588
    Epoch 4/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7529 - accuracy: 0.7672
    Epoch 5/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7175 - accuracy: 0.7756
    Epoch 6/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6925 - accuracy: 0.7896
    Epoch 7/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6693 - accuracy: 0.8036
    Epoch 8/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6412 - accuracy: 0.7896
    Epoch 9/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6256 - accuracy: 0.7938
    Epoch 10/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6206 - accuracy: 0.7882
    Epoch 11/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6026 - accuracy: 0.7840
    Epoch 12/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5945 - accuracy: 0.8022
    Epoch 13/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5918 - accuracy: 0.7980
    Epoch 14/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5789 - accuracy: 0.8079
    Epoch 15/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5633 - accuracy: 0.8163
    Epoch 16/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5736 - accuracy: 0.7938
    Epoch 17/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5659 - accuracy: 0.8065
    Epoch 18/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5560 - accuracy: 0.8079
    Epoch 19/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5461 - accuracy: 0.8149
    Epoch 20/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5474 - accuracy: 0.8022
    Epoch 21/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5375 - accuracy: 0.8177
    Epoch 22/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5337 - accuracy: 0.8205
    Epoch 23/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5240 - accuracy: 0.8261
    Epoch 24/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5182 - accuracy: 0.8205
    Epoch 25/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5231 - accuracy: 0.8177
    Epoch 26/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5188 - accuracy: 0.8233
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5178 - accuracy: 0.8261
    Epoch 28/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5142 - accuracy: 0.8205
    Epoch 29/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5006 - accuracy: 0.8233
    Epoch 30/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5000 - accuracy: 0.8275
    Epoch 31/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5066 - accuracy: 0.8275
    Epoch 32/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5202 - accuracy: 0.8149
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5065 - accuracy: 0.8163
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5025 - accuracy: 0.8261
    Epoch 35/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4974 - accuracy: 0.8261
    Epoch 36/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4912 - accuracy: 0.8289
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4991 - accuracy: 0.8191
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4982 - accuracy: 0.8219
    Epoch 39/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5012 - accuracy: 0.8191
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4878 - accuracy: 0.8331
    Epoch 41/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4956 - accuracy: 0.8107
    Epoch 42/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4909 - accuracy: 0.8289
    Epoch 43/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4995 - accuracy: 0.8219
    Epoch 44/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4976 - accuracy: 0.8191
    Epoch 45/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4864 - accuracy: 0.8359
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4960 - accuracy: 0.8261
    Epoch 47/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4868 - accuracy: 0.8303
    Epoch 48/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4985 - accuracy: 0.8317
    Epoch 49/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4926 - accuracy: 0.8177
    Epoch 50/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4821 - accuracy: 0.8345
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 2ms/step - loss: 0.9605 - accuracy: 0.6830
    Epoch 2/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8798 - accuracy: 0.7265
    Epoch 3/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7993 - accuracy: 0.7447
    Epoch 4/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7593 - accuracy: 0.7644
    Epoch 5/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6957 - accuracy: 0.7882
    Epoch 6/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6909 - accuracy: 0.7728
    Epoch 7/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6483 - accuracy: 0.8036
    Epoch 8/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6287 - accuracy: 0.8065
    Epoch 9/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6103 - accuracy: 0.8065
    Epoch 10/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6114 - accuracy: 0.7980
    Epoch 11/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5841 - accuracy: 0.8191
    Epoch 12/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5789 - accuracy: 0.8093
    Epoch 13/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5701 - accuracy: 0.8233
    Epoch 14/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5494 - accuracy: 0.8289
    Epoch 15/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5415 - accuracy: 0.8345
    Epoch 16/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5363 - accuracy: 0.8317
    Epoch 17/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5456 - accuracy: 0.8275
    Epoch 18/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5228 - accuracy: 0.8373
    Epoch 19/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5286 - accuracy: 0.8345
    Epoch 20/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5126 - accuracy: 0.8373
    Epoch 21/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5141 - accuracy: 0.8401
    Epoch 22/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5074 - accuracy: 0.8443
    Epoch 23/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4996 - accuracy: 0.8415
    Epoch 24/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5039 - accuracy: 0.8373
    Epoch 25/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5046 - accuracy: 0.8303
    Epoch 26/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4973 - accuracy: 0.8345
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4894 - accuracy: 0.8513
    Epoch 28/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4911 - accuracy: 0.8415
    Epoch 29/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4956 - accuracy: 0.8345
    Epoch 30/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4866 - accuracy: 0.8485
    Epoch 31/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4934 - accuracy: 0.8317
    Epoch 32/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4869 - accuracy: 0.8289
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4810 - accuracy: 0.8415
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4884 - accuracy: 0.8387
    Epoch 35/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4841 - accuracy: 0.8429
    Epoch 36/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4705 - accuracy: 0.8373
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4676 - accuracy: 0.8443
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4750 - accuracy: 0.8443
    Epoch 39/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4805 - accuracy: 0.8415
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4919 - accuracy: 0.8443
    Epoch 41/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4812 - accuracy: 0.8429
    Epoch 42/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4810 - accuracy: 0.8373
    Epoch 43/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4747 - accuracy: 0.8457
    Epoch 44/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4839 - accuracy: 0.8429
    Epoch 45/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4814 - accuracy: 0.8345
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4674 - accuracy: 0.8373
    Epoch 47/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4690 - accuracy: 0.8471
    Epoch 48/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4686 - accuracy: 0.8387
    Epoch 49/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4681 - accuracy: 0.8457
    Epoch 50/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4713 - accuracy: 0.8401
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 2ms/step - loss: 0.9941 - accuracy: 0.6157
    Epoch 2/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.9033 - accuracy: 0.7209
    Epoch 3/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8296 - accuracy: 0.7532
    Epoch 4/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7845 - accuracy: 0.7391
    Epoch 5/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7256 - accuracy: 0.7742
    Epoch 6/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7021 - accuracy: 0.7658
    Epoch 7/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6731 - accuracy: 0.7924
    Epoch 8/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6537 - accuracy: 0.8036
    Epoch 9/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6336 - accuracy: 0.7966
    Epoch 10/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6260 - accuracy: 0.7910
    Epoch 11/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6115 - accuracy: 0.7966
    Epoch 12/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5913 - accuracy: 0.8036
    Epoch 13/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5987 - accuracy: 0.7938
    Epoch 14/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5702 - accuracy: 0.8107
    Epoch 15/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5675 - accuracy: 0.8135
    Epoch 16/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5660 - accuracy: 0.8036
    Epoch 17/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5501 - accuracy: 0.8008
    Epoch 18/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5547 - accuracy: 0.7980
    Epoch 19/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5451 - accuracy: 0.8107
    Epoch 20/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5403 - accuracy: 0.8135
    Epoch 21/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5300 - accuracy: 0.8149
    Epoch 22/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5380 - accuracy: 0.8135
    Epoch 23/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5275 - accuracy: 0.8177
    Epoch 24/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5337 - accuracy: 0.8107
    Epoch 25/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5225 - accuracy: 0.8149
    Epoch 26/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5211 - accuracy: 0.8149
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5140 - accuracy: 0.8149
    Epoch 28/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5174 - accuracy: 0.8079
    Epoch 29/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5104 - accuracy: 0.8177
    Epoch 30/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5147 - accuracy: 0.8107
    Epoch 31/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5169 - accuracy: 0.8191
    Epoch 32/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5136 - accuracy: 0.8219
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5024 - accuracy: 0.8177
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5068 - accuracy: 0.8219
    Epoch 35/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5090 - accuracy: 0.8163
    Epoch 36/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5097 - accuracy: 0.8121
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5115 - accuracy: 0.8107
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5055 - accuracy: 0.8079
    Epoch 39/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5033 - accuracy: 0.8163
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5068 - accuracy: 0.8149
    Epoch 41/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5017 - accuracy: 0.8079
    Epoch 42/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4966 - accuracy: 0.8149
    Epoch 43/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4910 - accuracy: 0.8275
    Epoch 44/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4930 - accuracy: 0.8177
    Epoch 45/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4960 - accuracy: 0.8275
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5107 - accuracy: 0.7910
    Epoch 47/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4935 - accuracy: 0.8191
    Epoch 48/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5021 - accuracy: 0.8163
    Epoch 49/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4896 - accuracy: 0.8275
    Epoch 50/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4949 - accuracy: 0.8177
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    56/56 [==============================] - 1s 2ms/step - loss: 0.9021 - accuracy: 0.6891
    Epoch 2/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.7734 - accuracy: 0.7598
    Epoch 3/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.6917 - accuracy: 0.7912
    Epoch 4/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.6463 - accuracy: 0.7991
    Epoch 5/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.6037 - accuracy: 0.8171
    Epoch 6/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5872 - accuracy: 0.8126
    Epoch 7/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5859 - accuracy: 0.8137
    Epoch 8/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5768 - accuracy: 0.8171
    Epoch 9/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5440 - accuracy: 0.8193
    Epoch 10/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.5517 - accuracy: 0.8148
    Epoch 11/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5288 - accuracy: 0.8227
    Epoch 12/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5332 - accuracy: 0.8238
    Epoch 13/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.5175 - accuracy: 0.8159
    Epoch 14/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5212 - accuracy: 0.8193
    Epoch 15/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5075 - accuracy: 0.8227
    Epoch 16/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5080 - accuracy: 0.8238
    Epoch 17/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5064 - accuracy: 0.8238
    Epoch 18/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.5055 - accuracy: 0.8283
    Epoch 19/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4948 - accuracy: 0.8238
    Epoch 20/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4940 - accuracy: 0.8294
    Epoch 21/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4948 - accuracy: 0.8260
    Epoch 22/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4993 - accuracy: 0.8137
    Epoch 23/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4870 - accuracy: 0.8249
    Epoch 24/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4883 - accuracy: 0.8294
    Epoch 25/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4855 - accuracy: 0.8238
    Epoch 26/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4845 - accuracy: 0.8260
    Epoch 27/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4793 - accuracy: 0.8260
    Epoch 28/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4846 - accuracy: 0.8294
    Epoch 29/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.4795 - accuracy: 0.8328
    Epoch 30/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4824 - accuracy: 0.8294
    Epoch 31/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4822 - accuracy: 0.8238
    Epoch 32/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4869 - accuracy: 0.8238
    Epoch 33/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.4828 - accuracy: 0.8283
    Epoch 34/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4836 - accuracy: 0.8328
    Epoch 35/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4787 - accuracy: 0.8316
    Epoch 36/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4775 - accuracy: 0.8294
    Epoch 37/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4765 - accuracy: 0.8272
    Epoch 38/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4770 - accuracy: 0.8249
    Epoch 39/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4808 - accuracy: 0.8204
    Epoch 40/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.4835 - accuracy: 0.8227
    Epoch 41/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4770 - accuracy: 0.8215
    Epoch 42/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4673 - accuracy: 0.8361
    Epoch 43/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4776 - accuracy: 0.8283
    Epoch 44/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4747 - accuracy: 0.8316
    Epoch 45/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4692 - accuracy: 0.8294
    Epoch 46/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4723 - accuracy: 0.8294
    Epoch 47/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4760 - accuracy: 0.8294
    Epoch 48/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.4705 - accuracy: 0.8249
    Epoch 49/50
    56/56 [==============================] - 0s 3ms/step - loss: 0.4696 - accuracy: 0.8406
    Epoch 50/50
    56/56 [==============================] - 0s 2ms/step - loss: 0.4772 - accuracy: 0.8260


#### `TensorBoard`

`TensorBoard` is `TensorFlow`'s visualization toolkit. It is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. The code immediately below will allow us to use TensorBoard.

N.B. When we loaded the libraries, we loaded the TensorBoard notebook extension. (It is the last line of code in the first code chunk.)


```python
# Clear out any prior log data.
!rm -rf logs
# Be careful not to run this command if already have trained your model and you want to use TensorBoard.

# Sets up a timestamped log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(log_dir)


# The callback function, which will be called in the fit()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```

#### Fitting the optimal model and evaluating with `TensorBoaard`

Define the early stopping callback. Use your best values from grid serarch with `KerasClassifer` and finally fit the model.


```python
# Define the EarlyStopping object
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-8,
                           verbose=1, patience=5,
                           mode='min')

# Using KerasClassifier
classifier = KerasClassifier(build_fn = build_classifier,
                             optimizer=best_parameters['optimizer'],
                             batch_size=best_parameters['batch_size'],
                             epochs=best_parameters['epochs'])

# Fit the model with the tensorboard_callback
classifier.fit(X_train,
               y_train,
               verbose=1,
               callbacks=[early_stop, tensorboard_callback])


# Warning: If verbose = 0 (silent) or 2 (one line per epoch), then on TensorBoard's Graphs tab there will be an error.
# The other tabs in TensorBoard will still be function, but if you want the graphs then verbose needs to be 1 (progress bar).
```

    Epoch 1/50


    <ipython-input-9-df59d3a5d12d>:7: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      classifier = KerasClassifier(build_fn = build_classifier,


    31/56 [===============>..............] - ETA: 0s - loss: 0.9463 - accuracy: 0.6794 

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 1s 3ms/step - loss: 0.9091 - accuracy: 0.7015
    Epoch 2/50
    31/56 [===============>..............] - ETA: 0s - loss: 0.7721 - accuracy: 0.7742

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.7468 - accuracy: 0.7834
    Epoch 3/50
    55/56 [============================>.] - ETA: 0s - loss: 0.6733 - accuracy: 0.8023

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.6715 - accuracy: 0.8036
    Epoch 4/50
    56/56 [==============================] - ETA: 0s - loss: 0.6218 - accuracy: 0.8159

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.6218 - accuracy: 0.8159
    Epoch 5/50
    30/56 [===============>..............] - ETA: 0s - loss: 0.6092 - accuracy: 0.8021

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5963 - accuracy: 0.8103
    Epoch 6/50
    46/56 [=======================>......] - ETA: 0s - loss: 0.6028 - accuracy: 0.8043

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5909 - accuracy: 0.8070
    Epoch 7/50
    28/56 [==============>...............] - ETA: 0s - loss: 0.5823 - accuracy: 0.8036

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5633 - accuracy: 0.8103
    Epoch 8/50
    30/56 [===============>..............] - ETA: 0s - loss: 0.5651 - accuracy: 0.8146

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5517 - accuracy: 0.8126
    Epoch 9/50
    52/56 [==========================>...] - ETA: 0s - loss: 0.5403 - accuracy: 0.8089

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5404 - accuracy: 0.8103
    Epoch 10/50
    51/56 [==========================>...] - ETA: 0s - loss: 0.5341 - accuracy: 0.8100

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5272 - accuracy: 0.8137
    Epoch 11/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.5348 - accuracy: 0.8168

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5211 - accuracy: 0.8193
    Epoch 12/50
    30/56 [===============>..............] - ETA: 0s - loss: 0.5021 - accuracy: 0.8250

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5181 - accuracy: 0.8171
    Epoch 13/50
    51/56 [==========================>...] - ETA: 0s - loss: 0.5143 - accuracy: 0.8174

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5205 - accuracy: 0.8126
    Epoch 14/50
    31/56 [===============>..............] - ETA: 0s - loss: 0.5047 - accuracy: 0.8327

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5014 - accuracy: 0.8238
    Epoch 15/50
    31/56 [===============>..............] - ETA: 0s - loss: 0.5161 - accuracy: 0.8145

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.5034 - accuracy: 0.8215
    Epoch 16/50
    53/56 [===========================>..] - ETA: 0s - loss: 0.4981 - accuracy: 0.8278

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4931 - accuracy: 0.8294
    Epoch 17/50
    30/56 [===============>..............] - ETA: 0s - loss: 0.5010 - accuracy: 0.8250

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4975 - accuracy: 0.8283
    Epoch 18/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4679 - accuracy: 0.8341

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4963 - accuracy: 0.8193
    Epoch 19/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4944 - accuracy: 0.8276

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4910 - accuracy: 0.8193
    Epoch 20/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.5169 - accuracy: 0.8082

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4914 - accuracy: 0.8272
    Epoch 21/50
    31/56 [===============>..............] - ETA: 0s - loss: 0.4521 - accuracy: 0.8468

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4839 - accuracy: 0.8339
    Epoch 22/50
    31/56 [===============>..............] - ETA: 0s - loss: 0.5074 - accuracy: 0.8085

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4850 - accuracy: 0.8249
    Epoch 23/50
    50/56 [=========================>....] - ETA: 0s - loss: 0.4832 - accuracy: 0.8288

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4810 - accuracy: 0.8305
    Epoch 24/50
    56/56 [==============================] - ETA: 0s - loss: 0.4780 - accuracy: 0.8283

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4780 - accuracy: 0.8283
    Epoch 25/50
    50/56 [=========================>....] - ETA: 0s - loss: 0.4612 - accuracy: 0.8388

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4761 - accuracy: 0.8305
    Epoch 26/50
    30/56 [===============>..............] - ETA: 0s - loss: 0.4778 - accuracy: 0.8250

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4798 - accuracy: 0.8249
    Epoch 27/50
    56/56 [==============================] - ETA: 0s - loss: 0.4801 - accuracy: 0.8305

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4801 - accuracy: 0.8305
    Epoch 28/50
    56/56 [==============================] - ETA: 0s - loss: 0.4791 - accuracy: 0.8294

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4791 - accuracy: 0.8294
    Epoch 29/50
    56/56 [==============================] - ETA: 0s - loss: 0.4760 - accuracy: 0.8204

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4760 - accuracy: 0.8204
    Epoch 30/50
    50/56 [=========================>....] - ETA: 0s - loss: 0.4716 - accuracy: 0.8313

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4683 - accuracy: 0.8328
    Epoch 31/50
    56/56 [==============================] - ETA: 0s - loss: 0.4746 - accuracy: 0.8294

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4746 - accuracy: 0.8294
    Epoch 32/50
    55/56 [============================>.] - ETA: 0s - loss: 0.4775 - accuracy: 0.8398

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4750 - accuracy: 0.8418
    Epoch 33/50
    56/56 [==============================] - ETA: 0s - loss: 0.4777 - accuracy: 0.8260

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4777 - accuracy: 0.8260
    Epoch 34/50
    28/56 [==============>...............] - ETA: 0s - loss: 0.5228 - accuracy: 0.7902

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4703 - accuracy: 0.8316
    Epoch 35/50
    55/56 [============================>.] - ETA: 0s - loss: 0.4679 - accuracy: 0.8375

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4671 - accuracy: 0.8373
    Epoch 36/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.5095 - accuracy: 0.8017

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4723 - accuracy: 0.8294
    Epoch 37/50
    48/56 [========================>.....] - ETA: 0s - loss: 0.4797 - accuracy: 0.8229

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4717 - accuracy: 0.8305
    Epoch 38/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4905 - accuracy: 0.8103

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4753 - accuracy: 0.8283
    Epoch 39/50
    53/56 [===========================>..] - ETA: 0s - loss: 0.4683 - accuracy: 0.8231

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4667 - accuracy: 0.8260
    Epoch 40/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4614 - accuracy: 0.8362

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4772 - accuracy: 0.8260
    Epoch 41/50
    31/56 [===============>..............] - ETA: 0s - loss: 0.4725 - accuracy: 0.8286

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4639 - accuracy: 0.8361
    Epoch 42/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4473 - accuracy: 0.8578

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4709 - accuracy: 0.8350
    Epoch 43/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4324 - accuracy: 0.8491

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4771 - accuracy: 0.8148
    Epoch 44/50
    54/56 [===========================>..] - ETA: 0s - loss: 0.4715 - accuracy: 0.8241

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4701 - accuracy: 0.8249
    Epoch 45/50
    29/56 [==============>...............] - ETA: 0s - loss: 0.4711 - accuracy: 0.8405

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4765 - accuracy: 0.8384
    Epoch 46/50
    30/56 [===============>..............] - ETA: 0s - loss: 0.4721 - accuracy: 0.8313

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 2ms/step - loss: 0.4703 - accuracy: 0.8373
    Epoch 47/50
    50/56 [=========================>....] - ETA: 0s - loss: 0.4763 - accuracy: 0.8413

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4755 - accuracy: 0.8384
    Epoch 48/50
    55/56 [============================>.] - ETA: 0s - loss: 0.4710 - accuracy: 0.8284

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4691 - accuracy: 0.8305
    Epoch 49/50
    55/56 [============================>.] - ETA: 0s - loss: 0.4671 - accuracy: 0.8239

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 4ms/step - loss: 0.4650 - accuracy: 0.8249
    Epoch 50/50
    48/56 [========================>.....] - ETA: 0s - loss: 0.4673 - accuracy: 0.8320

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.4686 - accuracy: 0.8294





    <keras.callbacks.History at 0x7ff69d5cdf70>




```python
# Call TensorBoard within SaturnCloud [Comment this out if you are not in SaturnCloud]
import os
print(f"https://{os.getenv('SATURN_JUPYTER_BASE_DOMAIN')}/proxy/8000/")
%tensorboard --logdir logs/fit --port 8000 --bind_all
# This will generate a hyperlink. Click on that to open TensorBoard!
# (You'll see a 404 error below the link, just ignore that.)

# Call TensorBoard [Not in SaturnCloud, e.g. Colab]
# Uncomment the next time if you are not in SC
#%tensorboard --logdir logs/fit
```


    <IPython.core.display.Javascript object>


#### Results and Predictions

Calculate the predictions, save them as a csv, and print them.


```python
# Results

# Your code here (use more cells if you need to)
```


```python
# __SOLUTION__

# Results
preds = classifier.predict(test)
results = ids.assign(Survived=preds)
results['Survived'] = results['Survived'].apply(lambda row: 1 if row > 0.5 else 0)
results.to_csv('titanic_submission.csv',index=False)
results.head(20)

```

    14/14 [==============================] - 0s 1ms/step






  <div id="df-3feea7de-7960-4278-8911-aa9f0a552efc">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>903</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>904</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>905</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>906</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>907</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>908</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>909</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>910</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>911</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3feea7de-7960-4278-8911-aa9f0a552efc')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3feea7de-7960-4278-8911-aa9f0a552efc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3feea7de-7960-4278-8911-aa9f0a552efc');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Continue to tweak your model until you are happy with the results based on model evaluation.

## Conclusion

Now that you have the `TensorBoard` to help you look at your model, you can better understand how to tweak your model.

How do your predictions compare to what you did last time?

Remember that your "fancier" model may be less accurate... but that is okay if that is the case since we're trying to guard against variance with regularization techniques.

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

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard


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

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:37: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:50: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only


## Neural Network Model

### Building the model

#### Define the model as a pipeline

Let's use the data science pipeline for our neural network model.

As you are now using regularization to guard against high variance, i.e. overfitting the data, in the definition of the model below include *dropout* and/or *l2* regularization. Also, feel free to experiment with different activation functions.


```python
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


```python
# Grid Search
# You can play with optimizers, epochs, and batch sizes.

classifier = KerasClassifier(build_fn = build_classifier)
param_grid = dict(optimizer = ['Adam'],
                  epochs=[10, 20, 50],
                  batch_size=[16, 25, 32])
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy')
grid_result = grid.fit(X_train, y_train)
best_parameters = grid.best_params_
best_accuracy = grid.best_score_
```

    Epoch 1/10


    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      after removing the cwd from sys.path.


    45/45 [==============================] - 1s 2ms/step - loss: 0.9665 - accuracy: 0.6657
    Epoch 2/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.8291 - accuracy: 0.7725
    Epoch 3/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.7369 - accuracy: 0.7865
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6736 - accuracy: 0.8034
    Epoch 5/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.6317 - accuracy: 0.8118
    Epoch 6/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.6111 - accuracy: 0.8090
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5848 - accuracy: 0.8188
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5716 - accuracy: 0.8244
    Epoch 9/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5596 - accuracy: 0.8160
    Epoch 10/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5537 - accuracy: 0.8258
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 3ms/step - loss: 0.9857 - accuracy: 0.5989
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.8658 - accuracy: 0.6942
    Epoch 3/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.7847 - accuracy: 0.7209
    Epoch 4/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.7340 - accuracy: 0.7349
    Epoch 5/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.6821 - accuracy: 0.7672
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6397 - accuracy: 0.7896
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6108 - accuracy: 0.7966
    Epoch 8/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5897 - accuracy: 0.8079
    Epoch 9/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5813 - accuracy: 0.8036
    Epoch 10/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5541 - accuracy: 0.8219
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 3ms/step - loss: 0.9386 - accuracy: 0.6760
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.8219 - accuracy: 0.7405
    Epoch 3/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.7604 - accuracy: 0.7588
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6999 - accuracy: 0.7756
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6650 - accuracy: 0.7826
    Epoch 6/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.6398 - accuracy: 0.7980
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6067 - accuracy: 0.8079
    Epoch 8/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5999 - accuracy: 0.8065
    Epoch 9/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5756 - accuracy: 0.8121
    Epoch 10/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5615 - accuracy: 0.8177
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 2ms/step - loss: 0.9531 - accuracy: 0.6297
    Epoch 2/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.7960 - accuracy: 0.7798
    Epoch 3/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.7010 - accuracy: 0.8008
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6594 - accuracy: 0.8191
    Epoch 5/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.6214 - accuracy: 0.8149
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5956 - accuracy: 0.8275
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5829 - accuracy: 0.8275
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5563 - accuracy: 0.8275
    Epoch 9/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5379 - accuracy: 0.8205
    Epoch 10/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5459 - accuracy: 0.8275
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    45/45 [==============================] - 1s 3ms/step - loss: 0.9547 - accuracy: 0.6171
    Epoch 2/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.8303 - accuracy: 0.7265
    Epoch 3/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.7483 - accuracy: 0.7616
    Epoch 4/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6992 - accuracy: 0.7728
    Epoch 5/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6531 - accuracy: 0.7840
    Epoch 6/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6351 - accuracy: 0.7784
    Epoch 7/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.6046 - accuracy: 0.8022
    Epoch 8/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5900 - accuracy: 0.8008
    Epoch 9/10
    45/45 [==============================] - 0s 2ms/step - loss: 0.5943 - accuracy: 0.8036
    Epoch 10/10
    45/45 [==============================] - 0s 3ms/step - loss: 0.5739 - accuracy: 0.8036
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9352 - accuracy: 0.6629
    Epoch 2/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.8180 - accuracy: 0.7374
    Epoch 3/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7324 - accuracy: 0.7809
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6877 - accuracy: 0.7809
    Epoch 5/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6390 - accuracy: 0.8020
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6317 - accuracy: 0.7907
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5946 - accuracy: 0.8132
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5839 - accuracy: 0.8090
    Epoch 9/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5660 - accuracy: 0.8230
    Epoch 10/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5676 - accuracy: 0.8132
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5544 - accuracy: 0.8174
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5407 - accuracy: 0.8272
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5322 - accuracy: 0.8202
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5285 - accuracy: 0.8315
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5202 - accuracy: 0.8244
    Epoch 16/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5189 - accuracy: 0.8146
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5067 - accuracy: 0.8287
    Epoch 18/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5081 - accuracy: 0.8230
    Epoch 19/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5143 - accuracy: 0.8230
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4945 - accuracy: 0.8258
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 3ms/step - loss: 0.9123 - accuracy: 0.7055
    Epoch 2/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.7757 - accuracy: 0.7700
    Epoch 3/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6852 - accuracy: 0.7938
    Epoch 4/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6437 - accuracy: 0.7910
    Epoch 5/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.6322 - accuracy: 0.7826
    Epoch 6/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5847 - accuracy: 0.8050
    Epoch 7/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5777 - accuracy: 0.8121
    Epoch 8/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5672 - accuracy: 0.8065
    Epoch 9/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5573 - accuracy: 0.8289
    Epoch 10/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5403 - accuracy: 0.8149
    Epoch 11/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5360 - accuracy: 0.8191
    Epoch 12/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5266 - accuracy: 0.8205
    Epoch 13/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5210 - accuracy: 0.8219
    Epoch 14/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5083 - accuracy: 0.8373
    Epoch 15/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5060 - accuracy: 0.8345
    Epoch 16/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5040 - accuracy: 0.8359
    Epoch 17/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.4968 - accuracy: 0.8359
    Epoch 18/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.5006 - accuracy: 0.8275
    Epoch 19/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.4960 - accuracy: 0.8373
    Epoch 20/20
    45/45 [==============================] - 0s 4ms/step - loss: 0.4859 - accuracy: 0.8261
    6/6 [==============================] - 0s 5ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 3ms/step - loss: 0.9426 - accuracy: 0.6185
    Epoch 2/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.8268 - accuracy: 0.7391
    Epoch 3/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.7390 - accuracy: 0.7714
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6867 - accuracy: 0.7840
    Epoch 5/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6523 - accuracy: 0.7812
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6209 - accuracy: 0.8036
    Epoch 7/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5888 - accuracy: 0.8177
    Epoch 8/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5781 - accuracy: 0.7994
    Epoch 9/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5644 - accuracy: 0.8008
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5528 - accuracy: 0.8107
    Epoch 11/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5341 - accuracy: 0.8247
    Epoch 12/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5325 - accuracy: 0.8205
    Epoch 13/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5377 - accuracy: 0.8191
    Epoch 14/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5211 - accuracy: 0.8205
    Epoch 15/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5150 - accuracy: 0.8233
    Epoch 16/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5052 - accuracy: 0.8233
    Epoch 17/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5032 - accuracy: 0.8261
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4967 - accuracy: 0.8233
    Epoch 19/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.4951 - accuracy: 0.8247
    Epoch 20/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.4951 - accuracy: 0.8219
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9602 - accuracy: 0.5792
    Epoch 2/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.8247 - accuracy: 0.7209
    Epoch 3/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.7210 - accuracy: 0.7658
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6658 - accuracy: 0.7770
    Epoch 5/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6446 - accuracy: 0.7728
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6221 - accuracy: 0.7770
    Epoch 7/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6016 - accuracy: 0.7798
    Epoch 8/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5689 - accuracy: 0.8008
    Epoch 9/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5687 - accuracy: 0.7882
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5637 - accuracy: 0.8079
    Epoch 11/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5431 - accuracy: 0.8079
    Epoch 12/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5364 - accuracy: 0.8205
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5183 - accuracy: 0.8317
    Epoch 14/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5179 - accuracy: 0.8247
    Epoch 15/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.4979 - accuracy: 0.8443
    Epoch 16/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5004 - accuracy: 0.8331
    Epoch 17/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.4979 - accuracy: 0.8303
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4934 - accuracy: 0.8359
    Epoch 19/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.4897 - accuracy: 0.8415
    Epoch 20/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.4844 - accuracy: 0.8401
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/20
    45/45 [==============================] - 1s 2ms/step - loss: 0.9155 - accuracy: 0.6957
    Epoch 2/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.7854 - accuracy: 0.7742
    Epoch 3/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6970 - accuracy: 0.7826
    Epoch 4/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6646 - accuracy: 0.7784
    Epoch 5/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.6231 - accuracy: 0.7952
    Epoch 6/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.6047 - accuracy: 0.7840
    Epoch 7/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5860 - accuracy: 0.8050
    Epoch 8/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5785 - accuracy: 0.8079
    Epoch 9/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5788 - accuracy: 0.7980
    Epoch 10/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5481 - accuracy: 0.8121
    Epoch 11/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5459 - accuracy: 0.8008
    Epoch 12/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5272 - accuracy: 0.8261
    Epoch 13/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5452 - accuracy: 0.8191
    Epoch 14/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5290 - accuracy: 0.8107
    Epoch 15/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5228 - accuracy: 0.8205
    Epoch 16/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5230 - accuracy: 0.8163
    Epoch 17/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5204 - accuracy: 0.8163
    Epoch 18/20
    45/45 [==============================] - 0s 2ms/step - loss: 0.5238 - accuracy: 0.8149
    Epoch 19/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5285 - accuracy: 0.8079
    Epoch 20/20
    45/45 [==============================] - 0s 3ms/step - loss: 0.5146 - accuracy: 0.8163
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9412 - accuracy: 0.6587
    Epoch 2/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.7909 - accuracy: 0.7486
    Epoch 3/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.7206 - accuracy: 0.7528
    Epoch 4/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6782 - accuracy: 0.7781
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6479 - accuracy: 0.7781
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6221 - accuracy: 0.7879
    Epoch 7/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5995 - accuracy: 0.8020
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5852 - accuracy: 0.7823
    Epoch 9/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5678 - accuracy: 0.8090
    Epoch 10/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5652 - accuracy: 0.8048
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5496 - accuracy: 0.7978
    Epoch 12/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5449 - accuracy: 0.8034
    Epoch 13/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5368 - accuracy: 0.8034
    Epoch 14/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5282 - accuracy: 0.8202
    Epoch 15/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5211 - accuracy: 0.8132
    Epoch 16/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5228 - accuracy: 0.8146
    Epoch 17/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5055 - accuracy: 0.8287
    Epoch 18/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5095 - accuracy: 0.8146
    Epoch 19/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5131 - accuracy: 0.8216
    Epoch 20/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5144 - accuracy: 0.8272
    Epoch 21/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5043 - accuracy: 0.8216
    Epoch 22/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4930 - accuracy: 0.8202
    Epoch 23/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4895 - accuracy: 0.8301
    Epoch 24/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4939 - accuracy: 0.8301
    Epoch 25/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4907 - accuracy: 0.8230
    Epoch 26/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4849 - accuracy: 0.8258
    Epoch 27/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4914 - accuracy: 0.8174
    Epoch 28/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4774 - accuracy: 0.8258
    Epoch 29/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4894 - accuracy: 0.8202
    Epoch 30/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4936 - accuracy: 0.8244
    Epoch 31/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4769 - accuracy: 0.8399
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4764 - accuracy: 0.8287
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4872 - accuracy: 0.8160
    Epoch 34/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4826 - accuracy: 0.8272
    Epoch 35/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4719 - accuracy: 0.8216
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4753 - accuracy: 0.8371
    Epoch 37/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4827 - accuracy: 0.8287
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4797 - accuracy: 0.8287
    Epoch 39/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4704 - accuracy: 0.8385
    Epoch 40/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4711 - accuracy: 0.8132
    Epoch 41/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4683 - accuracy: 0.8399
    Epoch 42/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4742 - accuracy: 0.8258
    Epoch 43/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4611 - accuracy: 0.8371
    Epoch 44/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4753 - accuracy: 0.8287
    Epoch 45/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4728 - accuracy: 0.8413
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4641 - accuracy: 0.8455
    Epoch 47/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4667 - accuracy: 0.8343
    Epoch 48/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4795 - accuracy: 0.8202
    Epoch 49/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4608 - accuracy: 0.8315
    Epoch 50/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4685 - accuracy: 0.8427
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9313 - accuracy: 0.6606
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.8117 - accuracy: 0.7139
    Epoch 3/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.7340 - accuracy: 0.7588
    Epoch 4/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6790 - accuracy: 0.7868
    Epoch 5/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6502 - accuracy: 0.7854
    Epoch 6/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6296 - accuracy: 0.7840
    Epoch 7/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6126 - accuracy: 0.7910
    Epoch 8/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5978 - accuracy: 0.7910
    Epoch 9/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5779 - accuracy: 0.7924
    Epoch 10/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5711 - accuracy: 0.8008
    Epoch 11/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5408 - accuracy: 0.8065
    Epoch 12/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5244 - accuracy: 0.8191
    Epoch 13/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5396 - accuracy: 0.8050
    Epoch 14/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5293 - accuracy: 0.8149
    Epoch 15/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5129 - accuracy: 0.8233
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4988 - accuracy: 0.8289
    Epoch 17/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5046 - accuracy: 0.8163
    Epoch 18/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5002 - accuracy: 0.8359
    Epoch 19/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4965 - accuracy: 0.8247
    Epoch 20/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4929 - accuracy: 0.8345
    Epoch 21/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4976 - accuracy: 0.8205
    Epoch 22/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4976 - accuracy: 0.8191
    Epoch 23/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4923 - accuracy: 0.8317
    Epoch 24/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4899 - accuracy: 0.8331
    Epoch 25/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4834 - accuracy: 0.8289
    Epoch 26/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4744 - accuracy: 0.8303
    Epoch 27/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4781 - accuracy: 0.8373
    Epoch 28/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4808 - accuracy: 0.8219
    Epoch 29/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4811 - accuracy: 0.8261
    Epoch 30/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4887 - accuracy: 0.8261
    Epoch 31/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4827 - accuracy: 0.8345
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4630 - accuracy: 0.8359
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4686 - accuracy: 0.8345
    Epoch 34/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4768 - accuracy: 0.8317
    Epoch 35/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4655 - accuracy: 0.8373
    Epoch 36/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4737 - accuracy: 0.8359
    Epoch 37/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4685 - accuracy: 0.8303
    Epoch 38/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4655 - accuracy: 0.8345
    Epoch 39/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4714 - accuracy: 0.8275
    Epoch 40/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4637 - accuracy: 0.8261
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4672 - accuracy: 0.8345
    Epoch 42/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4671 - accuracy: 0.8387
    Epoch 43/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4649 - accuracy: 0.8247
    Epoch 44/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4705 - accuracy: 0.8317
    Epoch 45/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4717 - accuracy: 0.8331
    Epoch 46/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4663 - accuracy: 0.8289
    Epoch 47/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4667 - accuracy: 0.8359
    Epoch 48/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4685 - accuracy: 0.8429
    Epoch 49/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4560 - accuracy: 0.8401
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4636 - accuracy: 0.8303
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9673 - accuracy: 0.6648
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.8404 - accuracy: 0.7658
    Epoch 3/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.7533 - accuracy: 0.7602
    Epoch 4/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6997 - accuracy: 0.7742
    Epoch 5/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6650 - accuracy: 0.7784
    Epoch 6/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6310 - accuracy: 0.8008
    Epoch 7/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6080 - accuracy: 0.8050
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5949 - accuracy: 0.8036
    Epoch 9/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5705 - accuracy: 0.8149
    Epoch 10/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5664 - accuracy: 0.8050
    Epoch 11/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5548 - accuracy: 0.8233
    Epoch 12/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5543 - accuracy: 0.8008
    Epoch 13/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5388 - accuracy: 0.8163
    Epoch 14/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5352 - accuracy: 0.8219
    Epoch 15/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5235 - accuracy: 0.8163
    Epoch 16/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5246 - accuracy: 0.8247
    Epoch 17/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5161 - accuracy: 0.8205
    Epoch 18/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5181 - accuracy: 0.8331
    Epoch 19/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5071 - accuracy: 0.8233
    Epoch 20/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4986 - accuracy: 0.8247
    Epoch 21/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4958 - accuracy: 0.8191
    Epoch 22/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5051 - accuracy: 0.8247
    Epoch 23/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4930 - accuracy: 0.8289
    Epoch 24/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4839 - accuracy: 0.8289
    Epoch 25/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4907 - accuracy: 0.8331
    Epoch 26/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4933 - accuracy: 0.8331
    Epoch 27/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4785 - accuracy: 0.8443
    Epoch 28/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4934 - accuracy: 0.8289
    Epoch 29/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4794 - accuracy: 0.8359
    Epoch 30/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4849 - accuracy: 0.8233
    Epoch 31/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4771 - accuracy: 0.8317
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4795 - accuracy: 0.8331
    Epoch 33/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4713 - accuracy: 0.8373
    Epoch 34/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4846 - accuracy: 0.8247
    Epoch 35/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4862 - accuracy: 0.8415
    Epoch 36/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4857 - accuracy: 0.8233
    Epoch 37/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4847 - accuracy: 0.8261
    Epoch 38/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4865 - accuracy: 0.8219
    Epoch 39/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4816 - accuracy: 0.8317
    Epoch 40/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4785 - accuracy: 0.8289
    Epoch 41/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4695 - accuracy: 0.8387
    Epoch 42/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4851 - accuracy: 0.8303
    Epoch 43/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4817 - accuracy: 0.8247
    Epoch 44/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4768 - accuracy: 0.8387
    Epoch 45/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4699 - accuracy: 0.8373
    Epoch 46/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4773 - accuracy: 0.8359
    Epoch 47/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4730 - accuracy: 0.8373
    Epoch 48/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4802 - accuracy: 0.8289
    Epoch 49/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4770 - accuracy: 0.8303
    Epoch 50/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4798 - accuracy: 0.8275
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9254 - accuracy: 0.6732
    Epoch 2/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7945 - accuracy: 0.7602
    Epoch 3/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.7105 - accuracy: 0.7784
    Epoch 4/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6709 - accuracy: 0.7854
    Epoch 5/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6303 - accuracy: 0.7910
    Epoch 6/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5900 - accuracy: 0.8036
    Epoch 7/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5988 - accuracy: 0.7966
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5579 - accuracy: 0.8163
    Epoch 9/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5499 - accuracy: 0.8247
    Epoch 10/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5381 - accuracy: 0.8317
    Epoch 11/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5237 - accuracy: 0.8275
    Epoch 12/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5231 - accuracy: 0.8247
    Epoch 13/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5230 - accuracy: 0.8261
    Epoch 14/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5084 - accuracy: 0.8415
    Epoch 15/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4921 - accuracy: 0.8387
    Epoch 16/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4887 - accuracy: 0.8373
    Epoch 17/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4966 - accuracy: 0.8471
    Epoch 18/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4984 - accuracy: 0.8317
    Epoch 19/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4871 - accuracy: 0.8303
    Epoch 20/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4841 - accuracy: 0.8443
    Epoch 21/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4824 - accuracy: 0.8471
    Epoch 22/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4826 - accuracy: 0.8261
    Epoch 23/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4718 - accuracy: 0.8471
    Epoch 24/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4805 - accuracy: 0.8471
    Epoch 25/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4735 - accuracy: 0.8373
    Epoch 26/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4759 - accuracy: 0.8457
    Epoch 27/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4814 - accuracy: 0.8345
    Epoch 28/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4707 - accuracy: 0.8429
    Epoch 29/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4745 - accuracy: 0.8345
    Epoch 30/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4587 - accuracy: 0.8485
    Epoch 31/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4717 - accuracy: 0.8499
    Epoch 32/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4646 - accuracy: 0.8387
    Epoch 33/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4650 - accuracy: 0.8457
    Epoch 34/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4658 - accuracy: 0.8457
    Epoch 35/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4698 - accuracy: 0.8345
    Epoch 36/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4698 - accuracy: 0.8373
    Epoch 37/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4674 - accuracy: 0.8527
    Epoch 38/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4604 - accuracy: 0.8527
    Epoch 39/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4696 - accuracy: 0.8345
    Epoch 40/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4585 - accuracy: 0.8415
    Epoch 41/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.4581 - accuracy: 0.8471
    Epoch 42/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4694 - accuracy: 0.8373
    Epoch 43/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4646 - accuracy: 0.8429
    Epoch 44/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4603 - accuracy: 0.8499
    Epoch 45/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4626 - accuracy: 0.8499
    Epoch 46/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4542 - accuracy: 0.8443
    Epoch 47/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4532 - accuracy: 0.8499
    Epoch 48/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4614 - accuracy: 0.8401
    Epoch 49/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4648 - accuracy: 0.8415
    Epoch 50/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4608 - accuracy: 0.8401
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    45/45 [==============================] - 1s 2ms/step - loss: 0.9705 - accuracy: 0.6115
    Epoch 2/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.8393 - accuracy: 0.7139
    Epoch 3/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.7653 - accuracy: 0.7447
    Epoch 4/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6906 - accuracy: 0.7728
    Epoch 5/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6727 - accuracy: 0.7714
    Epoch 6/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6434 - accuracy: 0.7882
    Epoch 7/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.6064 - accuracy: 0.7966
    Epoch 8/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.6048 - accuracy: 0.8093
    Epoch 9/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5708 - accuracy: 0.8093
    Epoch 10/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5632 - accuracy: 0.8079
    Epoch 11/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5627 - accuracy: 0.8036
    Epoch 12/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5654 - accuracy: 0.8050
    Epoch 13/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5586 - accuracy: 0.8036
    Epoch 14/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5433 - accuracy: 0.8149
    Epoch 15/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5369 - accuracy: 0.8191
    Epoch 16/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5366 - accuracy: 0.8079
    Epoch 17/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5318 - accuracy: 0.8079
    Epoch 18/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5234 - accuracy: 0.8191
    Epoch 19/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5300 - accuracy: 0.8093
    Epoch 20/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5224 - accuracy: 0.8205
    Epoch 21/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5015 - accuracy: 0.8191
    Epoch 22/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5176 - accuracy: 0.8135
    Epoch 23/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5046 - accuracy: 0.8135
    Epoch 24/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5133 - accuracy: 0.8163
    Epoch 25/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5077 - accuracy: 0.8219
    Epoch 26/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5041 - accuracy: 0.8093
    Epoch 27/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4901 - accuracy: 0.8177
    Epoch 28/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5026 - accuracy: 0.8359
    Epoch 29/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4958 - accuracy: 0.8205
    Epoch 30/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5006 - accuracy: 0.8205
    Epoch 31/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4913 - accuracy: 0.8233
    Epoch 32/50
    45/45 [==============================] - 0s 2ms/step - loss: 0.5033 - accuracy: 0.8149
    Epoch 33/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4963 - accuracy: 0.8121
    Epoch 34/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4890 - accuracy: 0.8289
    Epoch 35/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4882 - accuracy: 0.8219
    Epoch 36/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.5020 - accuracy: 0.8191
    Epoch 37/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4989 - accuracy: 0.8177
    Epoch 38/50
    45/45 [==============================] - 0s 4ms/step - loss: 0.4870 - accuracy: 0.8233
    Epoch 39/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4951 - accuracy: 0.8135
    Epoch 40/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4871 - accuracy: 0.8289
    Epoch 41/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4934 - accuracy: 0.8247
    Epoch 42/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4873 - accuracy: 0.8345
    Epoch 43/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4995 - accuracy: 0.8219
    Epoch 44/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4918 - accuracy: 0.8149
    Epoch 45/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4796 - accuracy: 0.8261
    Epoch 46/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4842 - accuracy: 0.8275
    Epoch 47/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4926 - accuracy: 0.8177
    Epoch 48/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4861 - accuracy: 0.8191
    Epoch 49/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4900 - accuracy: 0.8219
    Epoch 50/50
    45/45 [==============================] - 0s 3ms/step - loss: 0.4769 - accuracy: 0.8289
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 3ms/step - loss: 0.9423 - accuracy: 0.6250
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8329 - accuracy: 0.7121
    Epoch 3/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7685 - accuracy: 0.7640
    Epoch 4/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7029 - accuracy: 0.7935
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6571 - accuracy: 0.8174
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6519 - accuracy: 0.7865
    Epoch 7/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6206 - accuracy: 0.8090
    Epoch 8/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5911 - accuracy: 0.8174
    Epoch 9/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5769 - accuracy: 0.8216
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5742 - accuracy: 0.8160
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 3ms/step - loss: 0.9639 - accuracy: 0.6031
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8711 - accuracy: 0.6718
    Epoch 3/10
    29/29 [==============================] - 0s 2ms/step - loss: 0.8072 - accuracy: 0.6985
    Epoch 4/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7531 - accuracy: 0.7433
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7164 - accuracy: 0.7475
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6742 - accuracy: 0.7826
    Epoch 7/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6345 - accuracy: 0.7924
    Epoch 8/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6251 - accuracy: 0.7784
    Epoch 9/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6015 - accuracy: 0.8093
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5816 - accuracy: 0.8036
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 3ms/step - loss: 0.9347 - accuracy: 0.6283
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8286 - accuracy: 0.7209
    Epoch 3/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7633 - accuracy: 0.7826
    Epoch 4/10
    29/29 [==============================] - 0s 4ms/step - loss: 0.7108 - accuracy: 0.7966
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6758 - accuracy: 0.7812
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6453 - accuracy: 0.8065
    Epoch 7/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6185 - accuracy: 0.8107
    Epoch 8/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6076 - accuracy: 0.8050
    Epoch 9/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5825 - accuracy: 0.8107
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.5679 - accuracy: 0.8205
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 3ms/step - loss: 0.9936 - accuracy: 0.6129
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8774 - accuracy: 0.7153
    Epoch 3/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8083 - accuracy: 0.7447
    Epoch 4/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7600 - accuracy: 0.7602
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7281 - accuracy: 0.7630
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6943 - accuracy: 0.7840
    Epoch 7/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6834 - accuracy: 0.7630
    Epoch 8/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6277 - accuracy: 0.8008
    Epoch 9/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6121 - accuracy: 0.8022
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6091 - accuracy: 0.8107
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    29/29 [==============================] - 1s 3ms/step - loss: 0.9701 - accuracy: 0.6031
    Epoch 2/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8673 - accuracy: 0.6844
    Epoch 3/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.8079 - accuracy: 0.7069
    Epoch 4/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7643 - accuracy: 0.7602
    Epoch 5/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7107 - accuracy: 0.7644
    Epoch 6/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.7103 - accuracy: 0.7644
    Epoch 7/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6689 - accuracy: 0.7840
    Epoch 8/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6545 - accuracy: 0.7910
    Epoch 9/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6402 - accuracy: 0.7868
    Epoch 10/10
    29/29 [==============================] - 0s 3ms/step - loss: 0.6205 - accuracy: 0.7938
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 3ms/step - loss: 0.9278 - accuracy: 0.6587
    Epoch 2/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8332 - accuracy: 0.7472
    Epoch 3/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7693 - accuracy: 0.7669
    Epoch 4/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7319 - accuracy: 0.7669
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7027 - accuracy: 0.7598
    Epoch 6/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6888 - accuracy: 0.7640
    Epoch 7/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6520 - accuracy: 0.7739
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6340 - accuracy: 0.7767
    Epoch 9/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6099 - accuracy: 0.8020
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5971 - accuracy: 0.7992
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5871 - accuracy: 0.7935
    Epoch 12/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5793 - accuracy: 0.7978
    Epoch 13/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5582 - accuracy: 0.8118
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5456 - accuracy: 0.8118
    Epoch 15/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5384 - accuracy: 0.8202
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5370 - accuracy: 0.8174
    Epoch 17/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5381 - accuracy: 0.8048
    Epoch 18/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5268 - accuracy: 0.8062
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5163 - accuracy: 0.8287
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5215 - accuracy: 0.8202
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 3ms/step - loss: 0.9929 - accuracy: 0.5554
    Epoch 2/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8831 - accuracy: 0.7167
    Epoch 3/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8098 - accuracy: 0.7391
    Epoch 4/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7524 - accuracy: 0.7475
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7003 - accuracy: 0.7686
    Epoch 6/20
    29/29 [==============================] - 0s 4ms/step - loss: 0.6665 - accuracy: 0.7784
    Epoch 7/20
    29/29 [==============================] - 0s 4ms/step - loss: 0.6551 - accuracy: 0.7714
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6125 - accuracy: 0.7868
    Epoch 9/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6068 - accuracy: 0.7854
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5966 - accuracy: 0.7868
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5720 - accuracy: 0.8050
    Epoch 12/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5671 - accuracy: 0.8163
    Epoch 13/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5611 - accuracy: 0.8079
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5454 - accuracy: 0.8289
    Epoch 15/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5538 - accuracy: 0.7980
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5350 - accuracy: 0.8219
    Epoch 17/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5370 - accuracy: 0.8149
    Epoch 18/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5153 - accuracy: 0.8191
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5234 - accuracy: 0.8163
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5103 - accuracy: 0.8149
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 3ms/step - loss: 0.9535 - accuracy: 0.6760
    Epoch 2/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8313 - accuracy: 0.7475
    Epoch 3/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7628 - accuracy: 0.7672
    Epoch 4/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7266 - accuracy: 0.7672
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6898 - accuracy: 0.7742
    Epoch 6/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6653 - accuracy: 0.7854
    Epoch 7/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6377 - accuracy: 0.7854
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6172 - accuracy: 0.7812
    Epoch 9/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6033 - accuracy: 0.7966
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5877 - accuracy: 0.8079
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5814 - accuracy: 0.8135
    Epoch 12/20
    29/29 [==============================] - 0s 4ms/step - loss: 0.5722 - accuracy: 0.8135
    Epoch 13/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5418 - accuracy: 0.8163
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5457 - accuracy: 0.8163
    Epoch 15/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5444 - accuracy: 0.8093
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5315 - accuracy: 0.8163
    Epoch 17/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5440 - accuracy: 0.8065
    Epoch 18/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5333 - accuracy: 0.8135
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5277 - accuracy: 0.8247
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5262 - accuracy: 0.8177
    6/6 [==============================] - 0s 5ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 3ms/step - loss: 0.9232 - accuracy: 0.6648
    Epoch 2/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8229 - accuracy: 0.7630
    Epoch 3/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7602 - accuracy: 0.7616
    Epoch 4/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7067 - accuracy: 0.7980
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6567 - accuracy: 0.8121
    Epoch 6/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6304 - accuracy: 0.8093
    Epoch 7/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6153 - accuracy: 0.8177
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5919 - accuracy: 0.8177
    Epoch 9/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5775 - accuracy: 0.8303
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5698 - accuracy: 0.8289
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5584 - accuracy: 0.8205
    Epoch 12/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5407 - accuracy: 0.8331
    Epoch 13/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5154 - accuracy: 0.8485
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5359 - accuracy: 0.8275
    Epoch 15/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5275 - accuracy: 0.8303
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5227 - accuracy: 0.8275
    Epoch 17/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5026 - accuracy: 0.8387
    Epoch 18/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5027 - accuracy: 0.8317
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5146 - accuracy: 0.8219
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5041 - accuracy: 0.8443
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/20
    29/29 [==============================] - 1s 3ms/step - loss: 0.9617 - accuracy: 0.6115
    Epoch 2/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8801 - accuracy: 0.6452
    Epoch 3/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.8139 - accuracy: 0.7083
    Epoch 4/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7605 - accuracy: 0.7307
    Epoch 5/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.7141 - accuracy: 0.7630
    Epoch 6/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6823 - accuracy: 0.7742
    Epoch 7/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6486 - accuracy: 0.7924
    Epoch 8/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6318 - accuracy: 0.8036
    Epoch 9/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6141 - accuracy: 0.7938
    Epoch 10/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.6115 - accuracy: 0.8036
    Epoch 11/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5935 - accuracy: 0.7812
    Epoch 12/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5862 - accuracy: 0.8008
    Epoch 13/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5802 - accuracy: 0.8093
    Epoch 14/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5854 - accuracy: 0.8050
    Epoch 15/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5542 - accuracy: 0.8121
    Epoch 16/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5589 - accuracy: 0.8135
    Epoch 17/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5573 - accuracy: 0.7966
    Epoch 18/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5524 - accuracy: 0.8177
    Epoch 19/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5514 - accuracy: 0.8149
    Epoch 20/20
    29/29 [==============================] - 0s 3ms/step - loss: 0.5452 - accuracy: 0.8050
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.9543 - accuracy: 0.6601
    Epoch 2/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8595 - accuracy: 0.7065
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8003 - accuracy: 0.7135
    Epoch 4/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.7381 - accuracy: 0.7486
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7131 - accuracy: 0.7472
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6874 - accuracy: 0.7598
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6690 - accuracy: 0.7683
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6405 - accuracy: 0.7851
    Epoch 9/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6270 - accuracy: 0.7907
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6123 - accuracy: 0.7865
    Epoch 11/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5901 - accuracy: 0.7963
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5878 - accuracy: 0.7907
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5785 - accuracy: 0.8132
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5710 - accuracy: 0.8090
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5392 - accuracy: 0.8301
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5542 - accuracy: 0.8104
    Epoch 17/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5548 - accuracy: 0.8076
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5523 - accuracy: 0.8174
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5208 - accuracy: 0.8258
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5222 - accuracy: 0.8244
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5166 - accuracy: 0.8258
    Epoch 22/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5269 - accuracy: 0.8132
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5237 - accuracy: 0.8230
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5076 - accuracy: 0.8315
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5150 - accuracy: 0.8216
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5170 - accuracy: 0.8287
    Epoch 27/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5058 - accuracy: 0.8174
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4979 - accuracy: 0.8287
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5049 - accuracy: 0.8413
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5024 - accuracy: 0.8315
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5034 - accuracy: 0.8258
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4966 - accuracy: 0.8385
    Epoch 33/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4963 - accuracy: 0.8202
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4863 - accuracy: 0.8329
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4810 - accuracy: 0.8371
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4882 - accuracy: 0.8371
    Epoch 37/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4942 - accuracy: 0.8329
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4908 - accuracy: 0.8343
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4867 - accuracy: 0.8315
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4797 - accuracy: 0.8287
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4839 - accuracy: 0.8427
    Epoch 42/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4856 - accuracy: 0.8315
    Epoch 43/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4809 - accuracy: 0.8174
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4812 - accuracy: 0.8301
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4765 - accuracy: 0.8329
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4805 - accuracy: 0.8272
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4799 - accuracy: 0.8343
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4848 - accuracy: 0.8174
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4773 - accuracy: 0.8343
    Epoch 50/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4806 - accuracy: 0.8258
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.9442 - accuracy: 0.7083
    Epoch 2/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8314 - accuracy: 0.7658
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7641 - accuracy: 0.7854
    Epoch 4/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7077 - accuracy: 0.8008
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6765 - accuracy: 0.8036
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6468 - accuracy: 0.7952
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6388 - accuracy: 0.7896
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6081 - accuracy: 0.8121
    Epoch 9/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5970 - accuracy: 0.8093
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5853 - accuracy: 0.8163
    Epoch 11/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5717 - accuracy: 0.8163
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5617 - accuracy: 0.8191
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5508 - accuracy: 0.8135
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5399 - accuracy: 0.8205
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5255 - accuracy: 0.8275
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5240 - accuracy: 0.8247
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5415 - accuracy: 0.8107
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5177 - accuracy: 0.8163
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5163 - accuracy: 0.8205
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5119 - accuracy: 0.8275
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5161 - accuracy: 0.8149
    Epoch 22/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5050 - accuracy: 0.8191
    Epoch 23/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5004 - accuracy: 0.8289
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4957 - accuracy: 0.8247
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5038 - accuracy: 0.8177
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4889 - accuracy: 0.8359
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4951 - accuracy: 0.8247
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4929 - accuracy: 0.8233
    Epoch 29/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4804 - accuracy: 0.8373
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4805 - accuracy: 0.8331
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4930 - accuracy: 0.8359
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4779 - accuracy: 0.8387
    Epoch 33/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4864 - accuracy: 0.8205
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4753 - accuracy: 0.8317
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4753 - accuracy: 0.8191
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4860 - accuracy: 0.8289
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4746 - accuracy: 0.8387
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4842 - accuracy: 0.8219
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4781 - accuracy: 0.8275
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4799 - accuracy: 0.8373
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4744 - accuracy: 0.8261
    Epoch 42/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4818 - accuracy: 0.8303
    Epoch 43/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4700 - accuracy: 0.8247
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4713 - accuracy: 0.8387
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4696 - accuracy: 0.8317
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4811 - accuracy: 0.8247
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4616 - accuracy: 0.8387
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4620 - accuracy: 0.8289
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4738 - accuracy: 0.8317
    Epoch 50/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4663 - accuracy: 0.8387
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.8979 - accuracy: 0.7153
    Epoch 2/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.8131 - accuracy: 0.7433
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7505 - accuracy: 0.7546
    Epoch 4/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7084 - accuracy: 0.7644
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6725 - accuracy: 0.7798
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6438 - accuracy: 0.7980
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6186 - accuracy: 0.8107
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6071 - accuracy: 0.7966
    Epoch 9/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.6025 - accuracy: 0.7868
    Epoch 10/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5947 - accuracy: 0.7854
    Epoch 11/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5646 - accuracy: 0.8289
    Epoch 12/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5635 - accuracy: 0.8036
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5541 - accuracy: 0.8107
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5546 - accuracy: 0.8036
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5480 - accuracy: 0.8022
    Epoch 16/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5445 - accuracy: 0.8121
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5203 - accuracy: 0.8247
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5272 - accuracy: 0.8219
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5160 - accuracy: 0.8205
    Epoch 20/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5202 - accuracy: 0.8233
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5158 - accuracy: 0.8205
    Epoch 22/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5013 - accuracy: 0.8205
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5071 - accuracy: 0.8303
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5078 - accuracy: 0.8373
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5038 - accuracy: 0.8247
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4936 - accuracy: 0.8233
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4979 - accuracy: 0.8247
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5028 - accuracy: 0.8247
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4892 - accuracy: 0.8261
    Epoch 30/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4906 - accuracy: 0.8205
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4956 - accuracy: 0.8303
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4878 - accuracy: 0.8331
    Epoch 33/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4904 - accuracy: 0.8233
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4842 - accuracy: 0.8317
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4905 - accuracy: 0.8303
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4902 - accuracy: 0.8205
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4889 - accuracy: 0.8205
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4816 - accuracy: 0.8303
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4859 - accuracy: 0.8247
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4813 - accuracy: 0.8317
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4887 - accuracy: 0.8261
    Epoch 42/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4837 - accuracy: 0.8289
    Epoch 43/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4836 - accuracy: 0.8317
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4855 - accuracy: 0.8317
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4947 - accuracy: 0.8247
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4803 - accuracy: 0.8303
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4800 - accuracy: 0.8331
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4786 - accuracy: 0.8317
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4775 - accuracy: 0.8345
    Epoch 50/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4802 - accuracy: 0.8247
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.9772 - accuracy: 0.5596
    Epoch 2/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.8422 - accuracy: 0.7532
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7654 - accuracy: 0.7560
    Epoch 4/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.6943 - accuracy: 0.7994
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6645 - accuracy: 0.7966
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6445 - accuracy: 0.7966
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6128 - accuracy: 0.8022
    Epoch 8/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5894 - accuracy: 0.8036
    Epoch 9/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5838 - accuracy: 0.8022
    Epoch 10/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5721 - accuracy: 0.8022
    Epoch 11/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5537 - accuracy: 0.8275
    Epoch 12/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5543 - accuracy: 0.8149
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5439 - accuracy: 0.8163
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5390 - accuracy: 0.8233
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5158 - accuracy: 0.8317
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5136 - accuracy: 0.8275
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5185 - accuracy: 0.8261
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5160 - accuracy: 0.8219
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5061 - accuracy: 0.8373
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5006 - accuracy: 0.8471
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4936 - accuracy: 0.8373
    Epoch 22/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4984 - accuracy: 0.8387
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4822 - accuracy: 0.8401
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4853 - accuracy: 0.8415
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4971 - accuracy: 0.8289
    Epoch 26/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4738 - accuracy: 0.8415
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4796 - accuracy: 0.8331
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4813 - accuracy: 0.8331
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4779 - accuracy: 0.8429
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4885 - accuracy: 0.8345
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4806 - accuracy: 0.8345
    Epoch 32/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4819 - accuracy: 0.8387
    Epoch 33/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4820 - accuracy: 0.8443
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4794 - accuracy: 0.8457
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4702 - accuracy: 0.8401
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4750 - accuracy: 0.8387
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4694 - accuracy: 0.8345
    Epoch 38/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4649 - accuracy: 0.8373
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4614 - accuracy: 0.8401
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4648 - accuracy: 0.8345
    Epoch 41/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4705 - accuracy: 0.8457
    Epoch 42/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4681 - accuracy: 0.8429
    Epoch 43/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4732 - accuracy: 0.8499
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4678 - accuracy: 0.8513
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4660 - accuracy: 0.8387
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4749 - accuracy: 0.8443
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4639 - accuracy: 0.8415
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4694 - accuracy: 0.8359
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4602 - accuracy: 0.8401
    Epoch 50/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4608 - accuracy: 0.8499
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    29/29 [==============================] - 1s 3ms/step - loss: 0.9491 - accuracy: 0.7055
    Epoch 2/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.8513 - accuracy: 0.7518
    Epoch 3/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7802 - accuracy: 0.7630
    Epoch 4/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7289 - accuracy: 0.7854
    Epoch 5/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.7106 - accuracy: 0.7742
    Epoch 6/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6946 - accuracy: 0.7770
    Epoch 7/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.6700 - accuracy: 0.7798
    Epoch 8/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.6480 - accuracy: 0.7812
    Epoch 9/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.6297 - accuracy: 0.8022
    Epoch 10/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.6157 - accuracy: 0.7868
    Epoch 11/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5997 - accuracy: 0.7980
    Epoch 12/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5944 - accuracy: 0.7910
    Epoch 13/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5826 - accuracy: 0.8065
    Epoch 14/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5752 - accuracy: 0.7980
    Epoch 15/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5729 - accuracy: 0.7910
    Epoch 16/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5610 - accuracy: 0.8079
    Epoch 17/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5579 - accuracy: 0.7910
    Epoch 18/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5455 - accuracy: 0.8107
    Epoch 19/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5385 - accuracy: 0.8036
    Epoch 20/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5370 - accuracy: 0.8135
    Epoch 21/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5292 - accuracy: 0.8050
    Epoch 22/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5376 - accuracy: 0.8065
    Epoch 23/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5169 - accuracy: 0.8149
    Epoch 24/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5241 - accuracy: 0.8093
    Epoch 25/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5231 - accuracy: 0.8205
    Epoch 26/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5172 - accuracy: 0.8050
    Epoch 27/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5059 - accuracy: 0.8107
    Epoch 28/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5212 - accuracy: 0.8121
    Epoch 29/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5100 - accuracy: 0.8233
    Epoch 30/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5073 - accuracy: 0.8205
    Epoch 31/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5044 - accuracy: 0.8177
    Epoch 32/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5147 - accuracy: 0.8135
    Epoch 33/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5050 - accuracy: 0.8163
    Epoch 34/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4988 - accuracy: 0.8289
    Epoch 35/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4989 - accuracy: 0.8177
    Epoch 36/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5008 - accuracy: 0.8177
    Epoch 37/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4934 - accuracy: 0.8247
    Epoch 38/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5025 - accuracy: 0.8191
    Epoch 39/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.5024 - accuracy: 0.8219
    Epoch 40/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4997 - accuracy: 0.8079
    Epoch 41/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5003 - accuracy: 0.8205
    Epoch 42/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.4934 - accuracy: 0.8093
    Epoch 43/50
    29/29 [==============================] - 0s 4ms/step - loss: 0.5004 - accuracy: 0.8149
    Epoch 44/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4948 - accuracy: 0.8163
    Epoch 45/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4852 - accuracy: 0.8121
    Epoch 46/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4941 - accuracy: 0.8149
    Epoch 47/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4845 - accuracy: 0.8177
    Epoch 48/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4816 - accuracy: 0.8135
    Epoch 49/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4918 - accuracy: 0.8219
    Epoch 50/50
    29/29 [==============================] - 0s 3ms/step - loss: 0.4996 - accuracy: 0.8135
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 3ms/step - loss: 0.9812 - accuracy: 0.5632
    Epoch 2/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.9060 - accuracy: 0.6713
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8488 - accuracy: 0.6840
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7968 - accuracy: 0.7051
    Epoch 5/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7496 - accuracy: 0.7570
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7049 - accuracy: 0.7893
    Epoch 7/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6687 - accuracy: 0.7978
    Epoch 8/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6409 - accuracy: 0.8188
    Epoch 9/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6214 - accuracy: 0.7992
    Epoch 10/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6010 - accuracy: 0.8090
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 3ms/step - loss: 0.9583 - accuracy: 0.6185
    Epoch 2/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8673 - accuracy: 0.7153
    Epoch 3/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.8073 - accuracy: 0.7447
    Epoch 4/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7625 - accuracy: 0.7658
    Epoch 5/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7244 - accuracy: 0.7714
    Epoch 6/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6862 - accuracy: 0.7798
    Epoch 7/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6740 - accuracy: 0.7952
    Epoch 8/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6556 - accuracy: 0.7616
    Epoch 9/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6252 - accuracy: 0.7854
    Epoch 10/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6162 - accuracy: 0.7896
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 3ms/step - loss: 1.0142 - accuracy: 0.6003
    Epoch 2/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.9035 - accuracy: 0.7433
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.8267 - accuracy: 0.7560
    Epoch 4/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7801 - accuracy: 0.7714
    Epoch 5/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7452 - accuracy: 0.7854
    Epoch 6/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7209 - accuracy: 0.7756
    Epoch 7/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6958 - accuracy: 0.7966
    Epoch 8/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6921 - accuracy: 0.7756
    Epoch 9/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6643 - accuracy: 0.7896
    Epoch 10/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6365 - accuracy: 0.8008
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 4ms/step - loss: 0.9438 - accuracy: 0.6802
    Epoch 2/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.8518 - accuracy: 0.7475
    Epoch 3/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7829 - accuracy: 0.7756
    Epoch 4/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7457 - accuracy: 0.7840
    Epoch 5/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7150 - accuracy: 0.7840
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6702 - accuracy: 0.8050
    Epoch 7/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6397 - accuracy: 0.8050
    Epoch 8/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6265 - accuracy: 0.8135
    Epoch 9/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.5996 - accuracy: 0.8163
    Epoch 10/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.5821 - accuracy: 0.8289
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/10
    23/23 [==============================] - 1s 3ms/step - loss: 0.9332 - accuracy: 0.6676
    Epoch 2/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.8529 - accuracy: 0.7419
    Epoch 3/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7864 - accuracy: 0.7546
    Epoch 4/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7652 - accuracy: 0.7714
    Epoch 5/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.7246 - accuracy: 0.7728
    Epoch 6/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.7053 - accuracy: 0.7658
    Epoch 7/10
    23/23 [==============================] - 0s 3ms/step - loss: 0.6798 - accuracy: 0.7812
    Epoch 8/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6464 - accuracy: 0.7896
    Epoch 9/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6384 - accuracy: 0.7826
    Epoch 10/10
    23/23 [==============================] - 0s 4ms/step - loss: 0.6237 - accuracy: 0.7896
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 0.9791 - accuracy: 0.6362
    Epoch 2/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8747 - accuracy: 0.7289
    Epoch 3/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8078 - accuracy: 0.7388
    Epoch 4/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7616 - accuracy: 0.7528
    Epoch 5/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7142 - accuracy: 0.7669
    Epoch 6/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6923 - accuracy: 0.7528
    Epoch 7/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6525 - accuracy: 0.7739
    Epoch 8/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6483 - accuracy: 0.7570
    Epoch 9/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6328 - accuracy: 0.7795
    Epoch 10/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6154 - accuracy: 0.7809
    Epoch 11/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5952 - accuracy: 0.7851
    Epoch 12/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5776 - accuracy: 0.7963
    Epoch 13/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5679 - accuracy: 0.8118
    Epoch 14/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5661 - accuracy: 0.8076
    Epoch 15/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5561 - accuracy: 0.8048
    Epoch 16/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5599 - accuracy: 0.8146
    Epoch 17/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5454 - accuracy: 0.8146
    Epoch 18/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5465 - accuracy: 0.8104
    Epoch 19/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5252 - accuracy: 0.8090
    Epoch 20/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5180 - accuracy: 0.8118
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 0.9463 - accuracy: 0.6536
    Epoch 2/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8439 - accuracy: 0.7714
    Epoch 3/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7725 - accuracy: 0.7854
    Epoch 4/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7156 - accuracy: 0.8289
    Epoch 5/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6791 - accuracy: 0.8121
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6576 - accuracy: 0.8135
    Epoch 7/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6365 - accuracy: 0.8149
    Epoch 8/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6050 - accuracy: 0.8205
    Epoch 9/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5912 - accuracy: 0.8135
    Epoch 10/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5842 - accuracy: 0.8050
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5717 - accuracy: 0.8177
    Epoch 12/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5610 - accuracy: 0.8233
    Epoch 13/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5510 - accuracy: 0.8191
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5468 - accuracy: 0.8303
    Epoch 15/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5473 - accuracy: 0.8177
    Epoch 16/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5375 - accuracy: 0.8149
    Epoch 17/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5290 - accuracy: 0.8261
    Epoch 18/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5278 - accuracy: 0.8219
    Epoch 19/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5219 - accuracy: 0.8289
    Epoch 20/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5205 - accuracy: 0.8261
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 4ms/step - loss: 1.0017 - accuracy: 0.6199
    Epoch 2/20
    23/23 [==============================] - 0s 5ms/step - loss: 0.8857 - accuracy: 0.7433
    Epoch 3/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8195 - accuracy: 0.7546
    Epoch 4/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7626 - accuracy: 0.7896
    Epoch 5/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7261 - accuracy: 0.7924
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6937 - accuracy: 0.7938
    Epoch 7/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6686 - accuracy: 0.7980
    Epoch 8/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6737 - accuracy: 0.7812
    Epoch 9/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6374 - accuracy: 0.8149
    Epoch 10/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6208 - accuracy: 0.8177
    Epoch 11/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6232 - accuracy: 0.8135
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5996 - accuracy: 0.8177
    Epoch 13/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5925 - accuracy: 0.8233
    Epoch 14/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5959 - accuracy: 0.8107
    Epoch 15/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5642 - accuracy: 0.8233
    Epoch 16/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5576 - accuracy: 0.8317
    Epoch 17/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5579 - accuracy: 0.8050
    Epoch 18/20
    23/23 [==============================] - 0s 5ms/step - loss: 0.5480 - accuracy: 0.8247
    Epoch 19/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5399 - accuracy: 0.8261
    Epoch 20/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5326 - accuracy: 0.8275
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 0.9951 - accuracy: 0.5610
    Epoch 2/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8732 - accuracy: 0.7532
    Epoch 3/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8037 - accuracy: 0.7756
    Epoch 4/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7433 - accuracy: 0.7924
    Epoch 5/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7194 - accuracy: 0.7952
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6819 - accuracy: 0.8065
    Epoch 7/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6721 - accuracy: 0.8022
    Epoch 8/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6406 - accuracy: 0.8093
    Epoch 9/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6121 - accuracy: 0.8121
    Epoch 10/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6165 - accuracy: 0.8022
    Epoch 11/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6046 - accuracy: 0.8065
    Epoch 12/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6021 - accuracy: 0.8022
    Epoch 13/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5702 - accuracy: 0.8261
    Epoch 14/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5748 - accuracy: 0.8177
    Epoch 15/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5588 - accuracy: 0.8247
    Epoch 16/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5514 - accuracy: 0.8219
    Epoch 17/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5478 - accuracy: 0.8359
    Epoch 18/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5571 - accuracy: 0.8135
    Epoch 19/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5452 - accuracy: 0.8205
    Epoch 20/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5209 - accuracy: 0.8331
    6/6 [==============================] - 0s 2ms/step
    Epoch 1/20
    23/23 [==============================] - 1s 3ms/step - loss: 1.0038 - accuracy: 0.5694
    Epoch 2/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.9012 - accuracy: 0.7265
    Epoch 3/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.8293 - accuracy: 0.7630
    Epoch 4/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.7716 - accuracy: 0.7854
    Epoch 5/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.7186 - accuracy: 0.8022
    Epoch 6/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.6874 - accuracy: 0.7882
    Epoch 7/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6757 - accuracy: 0.7854
    Epoch 8/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6598 - accuracy: 0.7770
    Epoch 9/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6207 - accuracy: 0.7980
    Epoch 10/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.6176 - accuracy: 0.8008
    Epoch 11/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5908 - accuracy: 0.8093
    Epoch 12/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5826 - accuracy: 0.8135
    Epoch 13/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5728 - accuracy: 0.8149
    Epoch 14/20
    23/23 [==============================] - 0s 3ms/step - loss: 0.5699 - accuracy: 0.8065
    Epoch 15/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5691 - accuracy: 0.8065
    Epoch 16/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5590 - accuracy: 0.8135
    Epoch 17/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5530 - accuracy: 0.8121
    Epoch 18/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5556 - accuracy: 0.7966
    Epoch 19/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5486 - accuracy: 0.8135
    Epoch 20/20
    23/23 [==============================] - 0s 4ms/step - loss: 0.5352 - accuracy: 0.8149
    6/6 [==============================] - 0s 4ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 3ms/step - loss: 0.9719 - accuracy: 0.5941
    Epoch 2/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.8869 - accuracy: 0.6980
    Epoch 3/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.8208 - accuracy: 0.7388
    Epoch 4/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7654 - accuracy: 0.7570
    Epoch 5/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7171 - accuracy: 0.7739
    Epoch 6/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6763 - accuracy: 0.7879
    Epoch 7/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6577 - accuracy: 0.7992
    Epoch 8/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6465 - accuracy: 0.7893
    Epoch 9/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6210 - accuracy: 0.8020
    Epoch 10/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6168 - accuracy: 0.7978
    Epoch 11/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5908 - accuracy: 0.8076
    Epoch 12/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5877 - accuracy: 0.8034
    Epoch 13/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5786 - accuracy: 0.7949
    Epoch 14/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5720 - accuracy: 0.8062
    Epoch 15/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5598 - accuracy: 0.8160
    Epoch 16/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5652 - accuracy: 0.8034
    Epoch 17/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5344 - accuracy: 0.8329
    Epoch 18/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5412 - accuracy: 0.8244
    Epoch 19/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5414 - accuracy: 0.8104
    Epoch 20/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5441 - accuracy: 0.8048
    Epoch 21/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5213 - accuracy: 0.8258
    Epoch 22/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5218 - accuracy: 0.8160
    Epoch 23/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5171 - accuracy: 0.8258
    Epoch 24/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5228 - accuracy: 0.8132
    Epoch 25/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5133 - accuracy: 0.8287
    Epoch 26/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5172 - accuracy: 0.8202
    Epoch 27/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5078 - accuracy: 0.8146
    Epoch 28/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4953 - accuracy: 0.8272
    Epoch 29/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5111 - accuracy: 0.8174
    Epoch 30/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4997 - accuracy: 0.8146
    Epoch 31/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5062 - accuracy: 0.8216
    Epoch 32/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5024 - accuracy: 0.8272
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4967 - accuracy: 0.8216
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4887 - accuracy: 0.8272
    Epoch 35/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5044 - accuracy: 0.8315
    Epoch 36/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4852 - accuracy: 0.8287
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4942 - accuracy: 0.8244
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4913 - accuracy: 0.8343
    Epoch 39/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4832 - accuracy: 0.8287
    Epoch 40/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4962 - accuracy: 0.8216
    Epoch 41/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4838 - accuracy: 0.8315
    Epoch 42/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4862 - accuracy: 0.8202
    Epoch 43/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4795 - accuracy: 0.8357
    Epoch 44/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4763 - accuracy: 0.8343
    Epoch 45/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4864 - accuracy: 0.8244
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4687 - accuracy: 0.8371
    Epoch 47/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4754 - accuracy: 0.8301
    Epoch 48/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4703 - accuracy: 0.8399
    Epoch 49/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4724 - accuracy: 0.8315
    Epoch 50/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4807 - accuracy: 0.8343
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 3ms/step - loss: 0.9273 - accuracy: 0.6830
    Epoch 2/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.8209 - accuracy: 0.7588
    Epoch 3/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7572 - accuracy: 0.7504
    Epoch 4/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7066 - accuracy: 0.7770
    Epoch 5/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6644 - accuracy: 0.7798
    Epoch 6/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6424 - accuracy: 0.7896
    Epoch 7/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6176 - accuracy: 0.7966
    Epoch 8/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6134 - accuracy: 0.8036
    Epoch 9/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6045 - accuracy: 0.8121
    Epoch 10/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5745 - accuracy: 0.8205
    Epoch 11/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5698 - accuracy: 0.8191
    Epoch 12/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5464 - accuracy: 0.8219
    Epoch 13/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5441 - accuracy: 0.8163
    Epoch 14/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5390 - accuracy: 0.8289
    Epoch 15/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5381 - accuracy: 0.8107
    Epoch 16/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5319 - accuracy: 0.8205
    Epoch 17/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5229 - accuracy: 0.8289
    Epoch 18/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5196 - accuracy: 0.8275
    Epoch 19/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5102 - accuracy: 0.8317
    Epoch 20/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5205 - accuracy: 0.8275
    Epoch 21/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5141 - accuracy: 0.8261
    Epoch 22/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5061 - accuracy: 0.8345
    Epoch 23/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5055 - accuracy: 0.8415
    Epoch 24/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5103 - accuracy: 0.8219
    Epoch 25/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4964 - accuracy: 0.8317
    Epoch 26/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5027 - accuracy: 0.8345
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4986 - accuracy: 0.8261
    Epoch 28/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4880 - accuracy: 0.8177
    Epoch 29/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4928 - accuracy: 0.8205
    Epoch 30/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4853 - accuracy: 0.8359
    Epoch 31/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4812 - accuracy: 0.8261
    Epoch 32/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4839 - accuracy: 0.8373
    Epoch 33/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4833 - accuracy: 0.8317
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4791 - accuracy: 0.8401
    Epoch 35/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.4832 - accuracy: 0.8107
    Epoch 36/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4654 - accuracy: 0.8373
    Epoch 37/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4720 - accuracy: 0.8331
    Epoch 38/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4821 - accuracy: 0.8429
    Epoch 39/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4754 - accuracy: 0.8331
    Epoch 40/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4782 - accuracy: 0.8345
    Epoch 41/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4823 - accuracy: 0.8289
    Epoch 42/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4747 - accuracy: 0.8219
    Epoch 43/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4721 - accuracy: 0.8457
    Epoch 44/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4833 - accuracy: 0.8261
    Epoch 45/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4757 - accuracy: 0.8373
    Epoch 46/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4629 - accuracy: 0.8443
    Epoch 47/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4711 - accuracy: 0.8345
    Epoch 48/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4713 - accuracy: 0.8387
    Epoch 49/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4708 - accuracy: 0.8275
    Epoch 50/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4707 - accuracy: 0.8387
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 4ms/step - loss: 0.9476 - accuracy: 0.6942
    Epoch 2/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.8394 - accuracy: 0.7518
    Epoch 3/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7641 - accuracy: 0.7784
    Epoch 4/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7226 - accuracy: 0.7910
    Epoch 5/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6889 - accuracy: 0.7854
    Epoch 6/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6662 - accuracy: 0.7938
    Epoch 7/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6417 - accuracy: 0.7980
    Epoch 8/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6203 - accuracy: 0.8107
    Epoch 9/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6029 - accuracy: 0.8065
    Epoch 10/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5859 - accuracy: 0.8191
    Epoch 11/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5859 - accuracy: 0.8036
    Epoch 12/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5597 - accuracy: 0.8219
    Epoch 13/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5597 - accuracy: 0.8233
    Epoch 14/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5610 - accuracy: 0.8121
    Epoch 15/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5531 - accuracy: 0.8149
    Epoch 16/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5382 - accuracy: 0.8163
    Epoch 17/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5338 - accuracy: 0.8261
    Epoch 18/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5316 - accuracy: 0.8191
    Epoch 19/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5243 - accuracy: 0.8191
    Epoch 20/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5258 - accuracy: 0.8177
    Epoch 21/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5129 - accuracy: 0.8345
    Epoch 22/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5092 - accuracy: 0.8387
    Epoch 23/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5019 - accuracy: 0.8331
    Epoch 24/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5137 - accuracy: 0.8219
    Epoch 25/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5071 - accuracy: 0.8359
    Epoch 26/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5017 - accuracy: 0.8331
    Epoch 27/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5000 - accuracy: 0.8359
    Epoch 28/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4990 - accuracy: 0.8233
    Epoch 29/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5084 - accuracy: 0.8317
    Epoch 30/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5052 - accuracy: 0.8261
    Epoch 31/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4969 - accuracy: 0.8275
    Epoch 32/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4913 - accuracy: 0.8317
    Epoch 33/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4876 - accuracy: 0.8401
    Epoch 34/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4886 - accuracy: 0.8275
    Epoch 35/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4857 - accuracy: 0.8219
    Epoch 36/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4880 - accuracy: 0.8261
    Epoch 37/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4908 - accuracy: 0.8275
    Epoch 38/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4883 - accuracy: 0.8331
    Epoch 39/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4905 - accuracy: 0.8345
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4824 - accuracy: 0.8401
    Epoch 41/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4796 - accuracy: 0.8275
    Epoch 42/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4835 - accuracy: 0.8317
    Epoch 43/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4778 - accuracy: 0.8261
    Epoch 44/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4690 - accuracy: 0.8359
    Epoch 45/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4794 - accuracy: 0.8205
    Epoch 46/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4777 - accuracy: 0.8261
    Epoch 47/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4744 - accuracy: 0.8317
    Epoch 48/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4814 - accuracy: 0.8401
    Epoch 49/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4730 - accuracy: 0.8317
    Epoch 50/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4741 - accuracy: 0.8331
    6/6 [==============================] - 0s 6ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 4ms/step - loss: 0.9669 - accuracy: 0.6732
    Epoch 2/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.8733 - accuracy: 0.7349
    Epoch 3/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.8078 - accuracy: 0.7475
    Epoch 4/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7582 - accuracy: 0.7798
    Epoch 5/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.7219 - accuracy: 0.7854
    Epoch 6/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7081 - accuracy: 0.7980
    Epoch 7/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6726 - accuracy: 0.8050
    Epoch 8/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6527 - accuracy: 0.8036
    Epoch 9/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6331 - accuracy: 0.8022
    Epoch 10/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.6010 - accuracy: 0.8079
    Epoch 11/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6145 - accuracy: 0.8036
    Epoch 12/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5949 - accuracy: 0.8163
    Epoch 13/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5753 - accuracy: 0.8191
    Epoch 14/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5834 - accuracy: 0.8065
    Epoch 15/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5631 - accuracy: 0.8149
    Epoch 16/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5659 - accuracy: 0.8163
    Epoch 17/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5462 - accuracy: 0.8289
    Epoch 18/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5348 - accuracy: 0.8247
    Epoch 19/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5374 - accuracy: 0.8275
    Epoch 20/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5304 - accuracy: 0.8275
    Epoch 21/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5241 - accuracy: 0.8219
    Epoch 22/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5119 - accuracy: 0.8331
    Epoch 23/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5133 - accuracy: 0.8233
    Epoch 24/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5146 - accuracy: 0.8317
    Epoch 25/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5030 - accuracy: 0.8387
    Epoch 26/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5065 - accuracy: 0.8205
    Epoch 27/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5036 - accuracy: 0.8289
    Epoch 28/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4965 - accuracy: 0.8289
    Epoch 29/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5050 - accuracy: 0.8359
    Epoch 30/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4970 - accuracy: 0.8233
    Epoch 31/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.4995 - accuracy: 0.8303
    Epoch 32/50
    23/23 [==============================] - 0s 6ms/step - loss: 0.4948 - accuracy: 0.8387
    Epoch 33/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4987 - accuracy: 0.8303
    Epoch 34/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4843 - accuracy: 0.8345
    Epoch 35/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4959 - accuracy: 0.8373
    Epoch 36/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4880 - accuracy: 0.8415
    Epoch 37/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4869 - accuracy: 0.8317
    Epoch 38/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4903 - accuracy: 0.8345
    Epoch 39/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4797 - accuracy: 0.8345
    Epoch 40/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4745 - accuracy: 0.8359
    Epoch 41/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4710 - accuracy: 0.8513
    Epoch 42/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4664 - accuracy: 0.8401
    Epoch 43/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4703 - accuracy: 0.8373
    Epoch 44/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4804 - accuracy: 0.8387
    Epoch 45/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4763 - accuracy: 0.8387
    Epoch 46/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.4676 - accuracy: 0.8457
    Epoch 47/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4713 - accuracy: 0.8359
    Epoch 48/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4674 - accuracy: 0.8443
    Epoch 49/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4705 - accuracy: 0.8415
    Epoch 50/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4716 - accuracy: 0.8457
    6/6 [==============================] - 0s 5ms/step
    Epoch 1/50
    23/23 [==============================] - 1s 4ms/step - loss: 1.0022 - accuracy: 0.5610
    Epoch 2/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.9009 - accuracy: 0.7223
    Epoch 3/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.8315 - accuracy: 0.7391
    Epoch 4/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7727 - accuracy: 0.7616
    Epoch 5/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7416 - accuracy: 0.7784
    Epoch 6/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.7012 - accuracy: 0.7882
    Epoch 7/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6843 - accuracy: 0.7812
    Epoch 8/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6508 - accuracy: 0.8107
    Epoch 9/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6391 - accuracy: 0.8093
    Epoch 10/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6125 - accuracy: 0.8093
    Epoch 11/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5971 - accuracy: 0.8022
    Epoch 12/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.6068 - accuracy: 0.8163
    Epoch 13/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5915 - accuracy: 0.8177
    Epoch 14/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5758 - accuracy: 0.8093
    Epoch 15/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5772 - accuracy: 0.7924
    Epoch 16/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5652 - accuracy: 0.8163
    Epoch 17/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5727 - accuracy: 0.7910
    Epoch 18/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5607 - accuracy: 0.8022
    Epoch 19/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5570 - accuracy: 0.8093
    Epoch 20/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5438 - accuracy: 0.8065
    Epoch 21/50
    23/23 [==============================] - 0s 5ms/step - loss: 0.5414 - accuracy: 0.8177
    Epoch 22/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5351 - accuracy: 0.8191
    Epoch 23/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5279 - accuracy: 0.8065
    Epoch 24/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5360 - accuracy: 0.8079
    Epoch 25/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5252 - accuracy: 0.8163
    Epoch 26/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5288 - accuracy: 0.8135
    Epoch 27/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5089 - accuracy: 0.8219
    Epoch 28/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5103 - accuracy: 0.8135
    Epoch 29/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5120 - accuracy: 0.8093
    Epoch 30/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5261 - accuracy: 0.8107
    Epoch 31/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5042 - accuracy: 0.8163
    Epoch 32/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5180 - accuracy: 0.8177
    Epoch 33/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5155 - accuracy: 0.8219
    Epoch 34/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5125 - accuracy: 0.8121
    Epoch 35/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5049 - accuracy: 0.8205
    Epoch 36/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5218 - accuracy: 0.8008
    Epoch 37/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5061 - accuracy: 0.8149
    Epoch 38/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5139 - accuracy: 0.8121
    Epoch 39/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5068 - accuracy: 0.8050
    Epoch 40/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.5035 - accuracy: 0.8121
    Epoch 41/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5063 - accuracy: 0.8149
    Epoch 42/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5121 - accuracy: 0.8121
    Epoch 43/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5078 - accuracy: 0.8121
    Epoch 44/50
    23/23 [==============================] - 0s 3ms/step - loss: 0.4970 - accuracy: 0.8261
    Epoch 45/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4952 - accuracy: 0.8247
    Epoch 46/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5015 - accuracy: 0.8177
    Epoch 47/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4872 - accuracy: 0.8303
    Epoch 48/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5030 - accuracy: 0.8205
    Epoch 49/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.4940 - accuracy: 0.8135
    Epoch 50/50
    23/23 [==============================] - 0s 4ms/step - loss: 0.5025 - accuracy: 0.8177
    6/6 [==============================] - 0s 3ms/step
    Epoch 1/10
    56/56 [==============================] - 1s 3ms/step - loss: 0.8653 - accuracy: 0.7059
    Epoch 2/10
    56/56 [==============================] - 0s 3ms/step - loss: 0.7390 - accuracy: 0.7789
    Epoch 3/10
    56/56 [==============================] - 0s 3ms/step - loss: 0.6768 - accuracy: 0.7991
    Epoch 4/10
    56/56 [==============================] - 0s 3ms/step - loss: 0.6496 - accuracy: 0.7946
    Epoch 5/10
    56/56 [==============================] - 0s 4ms/step - loss: 0.6234 - accuracy: 0.7912
    Epoch 6/10
    56/56 [==============================] - 0s 4ms/step - loss: 0.5906 - accuracy: 0.8070
    Epoch 7/10
    56/56 [==============================] - 0s 4ms/step - loss: 0.5670 - accuracy: 0.8081
    Epoch 8/10
    56/56 [==============================] - 0s 4ms/step - loss: 0.5597 - accuracy: 0.8126
    Epoch 9/10
    56/56 [==============================] - 0s 3ms/step - loss: 0.5486 - accuracy: 0.8070
    Epoch 10/10
    56/56 [==============================] - 0s 3ms/step - loss: 0.5406 - accuracy: 0.8081


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


# Warningr: If verbose = 0 (silent) or 2 (one line per epoch), then on TensorBoard's Graphs tab there will be an error.
# The other tabs in TensorBoard will still be function, but if you want the graphs then verbose needs to be 1 (progress bar).
```

    Epoch 1/10


    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:10: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      # Remove the CWD from sys.path while we load stuff.


    49/56 [=========================>....] - ETA: 0s - loss: 0.9109 - accuracy: 0.7143

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 1s 3ms/step - loss: 0.9040 - accuracy: 0.7116
    Epoch 2/10
    46/56 [=======================>......] - ETA: 0s - loss: 0.7803 - accuracy: 0.7527

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.7731 - accuracy: 0.7553
    Epoch 3/10
    44/56 [======================>.......] - ETA: 0s - loss: 0.7033 - accuracy: 0.7656

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.7049 - accuracy: 0.7677
    Epoch 4/10
    54/56 [===========================>..] - ETA: 0s - loss: 0.6698 - accuracy: 0.7523

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 4ms/step - loss: 0.6681 - accuracy: 0.7553
    Epoch 5/10
    41/56 [====================>.........] - ETA: 0s - loss: 0.6196 - accuracy: 0.7896

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.6206 - accuracy: 0.7845
    Epoch 6/10
    49/56 [=========================>....] - ETA: 0s - loss: 0.6108 - accuracy: 0.7819

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.6059 - accuracy: 0.7845
    Epoch 7/10
    43/56 [======================>.......] - ETA: 0s - loss: 0.5798 - accuracy: 0.7791

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5826 - accuracy: 0.7811
    Epoch 8/10
    41/56 [====================>.........] - ETA: 0s - loss: 0.5871 - accuracy: 0.7957

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 4ms/step - loss: 0.5668 - accuracy: 0.8047
    Epoch 9/10
    47/56 [========================>.....] - ETA: 0s - loss: 0.5466 - accuracy: 0.8098

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5462 - accuracy: 0.8081
    Epoch 10/10
    44/56 [======================>.......] - ETA: 0s - loss: 0.5384 - accuracy: 0.8082

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy


    56/56 [==============================] - 0s 3ms/step - loss: 0.5398 - accuracy: 0.8137





    <keras.callbacks.History at 0x7f731b0db390>




```python
# Call TensorBoard
%tensorboard --logdir logs/fit
```


    Reusing TensorBoard on port 6006 (pid 1193), started 0:14:58 ago. (Use '!kill 1193' to kill it.)



    <IPython.core.display.Javascript object>


#### Results and Predictions

Calcularte the predictions, save them as a csv, and print them.


```python
# results
preds = classifier.predict(test)
results = ids.assign(Survived=preds)
results['Survived'] = results['Survived'].apply(lambda row: 1 if row > 0.5 else 0)
results.to_csv('titanic_submission.csv',index=False)
results.head(20)
```

    14/14 [==============================] - 0s 2ms/step






  <div id="df-8e0a1879-0516-45d8-a4ef-016af282ca6e">
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
      <td>1</td>
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-8e0a1879-0516-45d8-a4ef-016af282ca6e')"
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
          document.querySelector('#df-8e0a1879-0516-45d8-a4ef-016af282ca6e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8e0a1879-0516-45d8-a4ef-016af282ca6e');
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

We'll continue with this for the next lesson when we learn about model regularization.

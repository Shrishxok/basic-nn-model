# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

### Importing Modules
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
```
### Authenticate & Create Dataframe using Data in Sheets
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MyMLData').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()
```
### Assign X and Y values
```
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X
y
```
### Normalize the values & Split the data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1
```
## Create a Neural Network & Train it:

Create the model
```
ai=Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])
```
Compile the model
```
ai.compile(optimizer='rmsprop',loss='mse')
```
Fit the model
```
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)
```
### Plot the Loss
```
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
```
### Evaluate the model
```
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
```
### Predict for some value
```
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information

<img width="247" alt="Screenshot 2023-08-21 at 3 35 08 PM" src="https://github.com/Shrishxok/basic-nn-model/assets/120294863/3d99bc02-584a-4fd9-81f4-6e580409bcee">


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="423" alt="Screenshot 2023-08-21 at 3 35 24 PM" src="https://github.com/Shrishxok/basic-nn-model/assets/120294863/c791e463-41a3-4f1f-b698-024a9213470a">


### Test Data Root Mean Squared Error

<img width="315" alt="261859908-f0d45a6a-504f-4ddc-afc5-2c99761597af" src="https://github.com/Shrishxok/basic-nn-model/assets/120294863/7fbb5c25-96ce-436c-b1b7-e1d0cadfc6b4">


### New Sample Data Prediction

<img width="408" alt="261859896-a7af5c03-2a78-4051-8f4b-cabf2bcbb365" src="https://github.com/Shrishxok/basic-nn-model/assets/120294863/c1d29a81-3044-4247-a17b-782f241da6cf">


## RESULT
Thus the neural network regression model for the given dataset is developed and executed successfully.

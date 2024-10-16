# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: THEJASWINI D
RegisterNumber:  212223110059
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data.csv')
dataset
```
![image](https://github.com/user-attachments/assets/7d66244d-9025-411a-babd-b6d3aa6badb3)
```
dataset = dataset.drop('sl_no',axis=1)
```
![image](https://github.com/user-attachments/assets/d11f8841-7bc6-4869-a61f-9148c1853645)
```
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
```
```
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
X
Y
```
![image](https://github.com/user-attachments/assets/1d7e3cd6-d272-498d-96dd-0f5c6a315a74)
```
theta = np.random.randn(X.shape[1])
y = Y
```
```
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
```
![image](https://github.com/user-attachments/assets/e9d65f1e-d870-444d-80ff-21347fb81acd)
```
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/e3723ec4-bc07-403a-8426-193ea911cae6)
```
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/b80d6a6a-89c6-4e7d-81b6-b176016d6b7e)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph 

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANDHIYA R
RegisterNumber: 212222230129
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.tail())
print(df.head())
df.info()
x=df.iloc[ :,:-1].values
print(x)
y=df.iloc[ :,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
Y_pred=reg.predict(x_test)
print(Y_pred)
print(y_test)
a=Y_pred-y_test
print(a)
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="green")
plt.title('Testing set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```
## Output:
### TRAINING SET
![image](https://github.com/SandhiyaR1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497571/8e9bcfca-9a88-479b-b772-05f8efc7829d)
### TESTING SET

![image](https://github.com/SandhiyaR1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497571/1ba0c0f7-b89f-4bcf-9103-d7fd3993c2ec)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('C:\\Users\Melywn Fernades\Downloads\out.csv')


#converting the dataset into numbers for correlation
t=['ORDERNUMBER','QUANTITYORDERED','PRICEEACH','ORDERLINENUMBER','SALES','ORDERDATE','QTR_ID','MONTH_ID','YEAR_ID','MSRP','POSTALCODE'] #already numbers 
l={}
a=0
del data["Unnamed: 0"]
for q in data:
    if q in t:
        pass
    else:
        for x in data[q]:
            if x in l.keys():
                pass
            else:
                a+=1
                l[x]=a
for c in l:
    data.replace(c, l[c], inplace=True)


co=['Blues']
def corel(y):
    plt.figure(figsize=(20,7))
    plt.title(y)
    sns.heatmap(data.corr(),annot=True, cmap=y)

array=[corel(col) for col in co[0:]]



data.info()


sns.pairplot(data)
plt.show()



##Statistical Details of the dataset
data.describe()


x=data.iloc[:, lambda df: [8,9,10,-8,-5,-1]] #8,9,-5,-1
y=data.iloc[:, lambda df: [4]]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#shapes of splitted data
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x_train,y_train)
y_pred=linreg.predict(x_test)
y_pred


from sklearn.metrics import r2_score
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy) 


testing_data_model_score = linreg.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)
training_data_model_score = linreg.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)


plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae)
print('Therefore our predicted value can be', mae, 'units more or less than the actual value.')  


predicted_sales = linreg.predict([[2, 2003,7,406,496,664]]) 
predicted_sales
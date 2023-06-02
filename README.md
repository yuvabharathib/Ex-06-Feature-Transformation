# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
## STEP 1:
Read the given Data

## STEP 2:
Clean the Data Set using Data Cleaning Process

## STEP 3:
Apply Feature Transformation techniques to all the features of the data set

## STEP 4:
Print the transformed features

# PPROGRAM:
```
Name: YUVABHARATHI.B
Reg No. 212222230181


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

# READ CSV FILES
df=pd.read_csv("/content/Data_to_Transform.csv")
df
# BASIC PROCESS
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()

# LOG TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

# RECIPROCAL TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# SQUARE ROOT TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# POWER TRANSFORMATION
# MODERATE POSITIVE SKEW
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

# QUANTILE TRANSFORMATION
# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUTPUT:
## Importing Libraries
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/9b0a8b8a-bea8-4ba6-93b0-791cc3d54cd8)


## Reading CSV File
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/e03df1d9-1523-461f-9d90-f65cc0c62fb9)


## Basic Process
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/94eaf388-e043-4f95-a739-a8fd5ee847c1)


## Before Transformation
## Highly Positive Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/0a04dcb1-8f2f-44b2-be6a-14466339901e)


## Highly Negative Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/6be222e0-51f8-4a94-a9a3-f86dcf60b183)



## Moderate Positive Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/3cacc6e4-ff29-4bd8-aa2a-463bca6476d7)


## Moderate Negative Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/46d8b5b8-3996-4888-b44a-18ee88c5e017)


## Log Transformation
## Highly Positive Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/b400b860-b005-4571-b490-71aec4cfa4fb)



## Moderate Positive Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/7a52d59b-be20-4753-96b5-9d71565d16bb)



## Reciprocal Transformation
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/27888f57-afd5-4017-ab1b-4a0c6b52edb9)


## Square Root Transformation
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/518f354f-10c7-4f22-be3c-df38fa4cd134)


## Power Transformation
## Moderate Positive Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/0983b745-079b-4985-b435-f9f23e3f7432)


## Moderate Negative Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/4e46622d-4ef2-4d8f-89e6-5d51d0ad0e00)


# Quantile Transformation
## Moderate Negative Skew
![image](https://github.com/yuvabharathib/Ex-06-Feature-Transformation/assets/113497404/c89a0b0a-07a6-4943-ab60-6e47c9788b32)


# RESULT:
Thus feature transformation is done for the given dataset.






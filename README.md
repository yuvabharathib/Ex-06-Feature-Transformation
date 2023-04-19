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

# PROGRAM
```
NAME: Yuvabharathi.B
REG NO: 212222230181


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT
![image](https://user-images.githubusercontent.com/113497404/232973134-3daf9536-3768-4192-bf65-c5721f086486.png)

![image](https://user-images.githubusercontent.com/113497404/232973178-d3403536-f480-4189-89c8-0611e5140f7f.png)

![image](https://user-images.githubusercontent.com/113497404/232973225-79215155-217b-4f92-98aa-ff7805b9abb0.png)

![image](https://user-images.githubusercontent.com/113497404/232973262-8aaaf9c1-c731-41a7-88ba-a7dd324c86d1.png)

![image](https://user-images.githubusercontent.com/113497404/232973314-f0ea4c52-aa43-4092-9ad7-3d3697a32d2c.png)

![image](https://user-images.githubusercontent.com/113497404/232973380-37550a29-c2c4-4cd4-a411-5264767fefe2.png)

![image](https://user-images.githubusercontent.com/113497404/232973406-54c415ac-9b3c-4e80-8c0f-751da80a354c.png)

![image](https://user-images.githubusercontent.com/113497404/232973441-211b4acb-2a73-4e94-811b-d6755bd95cff.png)

![image](https://user-images.githubusercontent.com/113497404/232973476-1e02dacf-df06-4368-9bb5-80d1dc22b38b.png)

![image](https://user-images.githubusercontent.com/113497404/232973500-fca227bb-7f93-486f-9cdb-87d175eb1edb.png)


# RESULT
Thus feature transformation is done for the given dataset.


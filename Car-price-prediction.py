#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns 
sns.set()


# ## Loading the dataset

# In[65]:


raw_data = pd.read_csv(r'C:\Users\swagat\Downloads\CAR DETAILS FROM CAR DEKHO.csv')


# ## Read about dataset

# In[66]:


raw_data.head()


# In[67]:


raw_data.describe(include = 'all')


# In[68]:


raw_data.info()


# ## Drop the variable that is not necessary for our prediction

# In[69]:


data = raw_data.drop(['name'], axis = 1)
data 


# ## Converting 'year' to 'age' column

# In[70]:


max = data['year'].max()
age = data['year'].apply(lambda x:(max+1)-x)
data.drop(['year'], axis = 1, inplace = True)
data ['age'] = age
data


# ## Exploring the PDF (Probability distributive function)

# In[71]:


sns.displot(data['selling_price'])
plt.show()


# In[72]:


sns.displot(data['km_driven'])
plt.show()

Removing the outliers
# In[73]:


q = data['km_driven'].quantile(0.99)
data = data[data['km_driven']<q]
data.describe(include = 'all')


# In[74]:


sns.displot(data['age'])
plt.show()


# ## Checking the OLS assumption

# In[75]:


f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))
ax1.scatter(data['age'],data['selling_price'])
ax1.set_title('Price and age')
ax2.scatter(data['km_driven'],data['selling_price'])
ax2.set_title('Price and km driven')

plt.show()


# In[76]:


log_price = np.log(data['selling_price'])
data ['log_price'] = log_price
data 


# In[77]:


f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))
ax1.scatter(data['age'],data['log_price'])
ax1.set_title('Price and age')
ax2.scatter(data['km_driven'],data['log_price'])
ax2.set_title('Price and km driven')

plt.show()


# In[78]:


data.isnull().sum()


# In[79]:


data.columns.values


# ## Calculating VIF to check multicollinearity

# In[80]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data[['km_driven','age']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns


# In[81]:


vif


# In[82]:


data = pd.get_dummies(data, drop_first=True) 


# In[83]:


data


# In[84]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data[[ 'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual', 'seller_type_Trustmark Dealer', 'transmission_Manual', 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns


# In[85]:


vif


# # Linear Regression

# ## Setting dependent and independent variables

# In[86]:


target = data['log_price']
inputs = data.drop(['log_price'], axis = 1)


# ## Feature Scalling

# In[87]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[88]:


inputs_scaled = scaler.transform(inputs)
inputs_scaled


# In[89]:


## Split train-test data


# In[90]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state = 365)


# In[91]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[92]:


y_hat = reg.predict(x_train)


# In[93]:


plt.scatter(y_train,y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)',size =18)
plt.xlim(8,20)
plt.ylim(8,20)
plt.show()


# In[94]:


sns.displot(y_train - y_hat, kind='kde')
plt.title('Residual PDF', size = 18)
plt.show()


# In[95]:


reg.score(x_train, y_train)


# In[96]:


reg.intercept_


# In[97]:


reg.coef_


# In[98]:


reg.summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])
reg.summary ['Weights'] = reg.coef_
reg.summary


# ## Prediction

# In[99]:


y_pred = reg.predict(x_test)


# In[100]:


plt.scatter(y_test,y_pred, alpha = 0.2)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)',size =18)
plt.xlim(8,20)
plt.ylim(8,20)
plt.show()


# In[101]:


df_pf = pd.DataFrame(np.exp(y_pred), columns=['Prediction']) #performance
df_pf.head()


# In[102]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[103]:


y_test = y_test.reset_index(drop=True)


# In[104]:


y_test.head()


# In[105]:


df_pf['Target'] = np.exp(y_test)
df_pf


# ## Difference between Target and predicted value

# In[106]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[107]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf.describe()


# In[ ]:





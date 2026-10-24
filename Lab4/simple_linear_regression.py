#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('Salary_dataset.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df.head()


# In[2]:


from sklearn.model_selection import train_test_split
X=df[['YearsExperience']]
y=df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[3]:


X_test


# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[5]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[6]:


print(model.coef_)
print(model.intercept_)


# In[7]:


y_predict = model.predict(X_test)
y_predict


# In[8]:


slopes = model.coef_

# 2. Get the intercept
intercept = model.intercept_


# In[9]:


X_test


# In[10]:


slopes


# In[11]:


intercept


# In[12]:


results = pd.DataFrame({
    'YearsExperience': X_test['YearsExperience'].values,
    'Actual Salary': y_test.values,
    'Predicted Salary': y_predict
})
print(results)


# In[13]:


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test["YearsExperience"],
    y_test,
    label="Actual Values"
)

sorted_indices = np.argsort(X_test["YearsExperience"].values)

X_sorted = X_test["YearsExperience"].values[sorted_indices]
y_pred_sorted = y_predict[sorted_indices]

plt.plot(
    X_sorted,
    y_pred_sorted,
    label="Regression Line"
)

plt.title("Simple Linear Regression")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()


# In[14]:


mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

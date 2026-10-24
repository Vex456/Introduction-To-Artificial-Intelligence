#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('house_price_regression_dataset.csv')
df.head()


# In[2]:


from sklearn.model_selection import train_test_split
X=df[['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built','Lot_Size','Garage_Size','Neighborhood_Quality']]
y=df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[3]:


(X_test)


# In[4]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[5]:


print(model.coef_)


# In[6]:


print(model.intercept_)


# In[7]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the test set
y_predict = model.predict(X_test)

# Actual vs Predicted comparison
results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_predict
})
print('--- Sample Predictions vs Actual ---')
display(results.head(10))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

print(f'Model Evaluation Metrics:')
print(f'Mean Absolute Error (MAE):      ${mae:,.2f}')
print(f'Mean Squared Error (MSE):       ${mse:,.2f}')
print(f'Root Mean Squared Error (RMSE): ${rmse:,.2f}')
print(f'R-squared (R2 Score):           {r2:.4f}')

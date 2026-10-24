#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

data_path = 'diabetes.csv'
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')
df.head()


# In[2]:


from sklearn.model_selection import train_test_split
# Prepare features and target variable
X = df[['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction']]
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



# In[3]:


X_test


# In[4]:


# Weights
print(model.coef_)

# Constant / intercept
print(model.intercept_)


# In[5]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy:.4f}')


# In[6]:


y_pred


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Generate the confusion matrix and visualize using heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# In[8]:


import pickle

# Sample patient feature values: Age=21, BMI=35.7, BloodPressure=75.0, Insulin=102.5, Glucose=112, DiabetesPedigreeFunction=0.148
sample_patient = pd.DataFrame([{
    'Age': 21,
    'BMI': 35.7,
    'BloodPressure': 75.0,
    'Insulin': 102.5,
    'Glucose': 112,
    'DiabetesPedigreeFunction': 0.148
}])[X.columns]

sample_pred = model.predict(sample_patient)[0]
sample_prob = model.predict_proba(sample_patient)[0][1]

print('Sample Patient Features:')
display(sample_patient)
print(f'Prediction: {"Diabetic" if sample_pred == 1 else "Non-Diabetic"} (Class {sample_pred})')
print(f'Probability of Diabetes: {sample_prob:.4f}')

# Persist model for deployment (LAB06)
with open('My_Model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Saved trained model to My_Model.pkl for LAB06 web application.')

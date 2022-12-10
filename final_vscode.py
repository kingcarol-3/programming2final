#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import streamlit as st
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# #### Q1 Import Data 

# In[2]:


s = pd.read_csv("social_media_usage.csv") #read in dataset & name it s

# In[6]:


#function that takes in x makes it 1 or 0
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x 


# ***

# #### Q3 Create & Clean DataFrame for Model

#Clean variables for new dataframe
ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9,np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8,np.nan,s["educ2"]),
    "parent": np.where(s["par"] > 2,np.nan, clean_sm(s["par"])),
    "married": np.where(s["marital"] > 6, np.nan, clean_sm(s["marital"])),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
    })


#drop any missing values
ss = ss.dropna()


# #### Dataset = SS
# + Target Variable: LinkedIn User (sm_li)
#     + 1 = LinkedIn User
#     + 0 = Not LinkedIn User
# + Features:
#     + income (levels 1 - 9)
#     + education (levels 1 - 8)
#     + parent (binary)
#     + married (binary)
#     + female (binary)
#     + age (1 - 97)


# #### Q4 Create Target Variable and Feature Set

# In[17]:


#Create Target Vector (y) & Feature Set (x)
y = ss["sm_li"]
x = ss[["income", "education", "married", 
        "parent", "female", "age"]]


# ***

# #### Q5 Split Data into Testing and Training Sets

# In[18]:


#Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 123)


# Training Data:
#  + **x_train** contains 80% of the features dataset and will be used to train the machine learning model
#  + **y_train** contains 80% of the target variable and will be used with the x_train data to develop the machine learning model
#  
# Testing Data:
#  + **x_test** contains 20% of the data from the features and will be used to evaluate the model on "untouched data"
#  + **y_test** contains 20% of the target variable and will be used to produce performance metrics by comparing these values to the predicted y values.

# ***

# #### Q6 Train Model

# In[20]:


#initiate a logistic regression model
lr = LogisticRegression(class_weight = "balanced")


# In[21]:


#Fit model with the training data
lr.fit(x_train, y_train)




# #### Q10 Use the Model to Make Predictions

# In[30]:
#Title
img = Image.open('LinkedIn_Logo_2019.png')
st.image(img)
st.markdown("# LinkedIn User Prediction Machine")
st.markdown("#### Created by Caroline King")


st.markdown("### Please fill out the information below to predict if you are a LinkedIn User.")
st.markdown("Note: All information will remain private and will not be stored.")

#income
inc = st.selectbox("Income",
            options = ["Less than $10,000",
                        "$10,000 to $19,000",
                        "$20,000 to $29,000",
                        "$30,000 to $39,000",
                        "$40,000 to $49,000",
                        "$50,000 to $74,000",
                        "$75,000 to $99,000",
                        "$100,000 to $149,000",
                        "$150,000 or more"
                        ])

if inc == "Less than $10,000":
    inc = 1
elif inc == "$10,000 to $19,000":
    inc = 2
elif inc == "$20,000 to $29,000":
    inc = 3
elif inc == "$30,000 to $39,000":
    inc = 4
elif inc == "$40,000 to $49,000":
    inc = 5
elif inc == "$50,000 to $74,000":
    inc = 6
elif inc == "$75,000 to $99,000":
    inc = 7
elif inc == "$100,000 to $149,000":
    inc = 8
else:
    inc = 9

#Education
educ = st.selectbox("Education",
        options = ["Less than High School",
                    "High School Incomplete",
                    "High School Graduate",
                    "Some College, No Degree",
                    "Two-Year Associate Degree",
                    "Four-Year College or University Degree",
                    "Some Postgraduate Schooling",
                    "Postgraduate or Professional Degree"
                    ])
if educ == "Less than High School":
    educ = 1
elif educ == "High School Incomplete":
    educ = 2
elif educ == "High School Graduate":
    educ = 3
elif educ == "Some College, No Degree":
    educ = 4
elif educ == "Two-Year Associate Degree":
    educ = 5
elif educ == "Four-Year College or University Degree":
    educ = 6
elif educ == "Some Postgraduate Schooling":
    educ = 7
else:
    educ = 8

#Parent
par = st.selectbox("Are you a parent?",
                options = ["Yes",
                            "No"])
if par == "Yes":
    par = 1
else:
    par = 0

#Married
mar = st.selectbox("Are you married?",
                options = ["Yes",
                            "No"])
if mar == "Yes":
    mar = 1
else:
    mar = 0

#Gender
gen = st.selectbox("Do you identify as female?",
                options = ["Yes",
                            "No"])
if gen == "Yes":
    gen = 1
else: 
    gen = 0

#Age
age = st.slider(label = "Please select your age (1 - 97 years old)",
                min_value = 1,
                max_value = 97,
                value = 42)


# In[33]:


#Predicting if the above user uses LinkedIn

#Prediction Button
if st.button("Click Here for Results"):
        person =[inc, educ, par, mar, gen, age]
        predicted_class = lr.predict([person])
        user_pred = np.where(predicted_class == 1, "a LinkedIn User", "Not a LinkedIn User")
        probs = lr.predict_proba([person])
        st.markdown(f" ##### Based on this model, there is a {round(100*probs[0][1],2)}% chance that you are a LinkedIn User.")
        st.markdown(f" ##### This machine learning model predicts that you are: {user_pred}.")
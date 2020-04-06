#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Student Name : Carolina Duque Chopitea 
# Cohort       :Divisadero 4



################################################################################
# Import Packages
################################################################################

import pandas as pd                                   # data science essentials
import numpy as np                                    # Numeric Python
import matplotlib.pyplot as plt                       # data viz
import seaborn as sns                                 # Enhance data viz   



from sklearn.model_selection import train_test_split  # Train Tests Split data 
from sklearn.metrics import roc_auc_score             # AUC curve

# libraries for classification trees
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.tree import export_graphviz             # exports graphics
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
import pydotplus        





################################################################################
# Load Data
################################################################################




original_df = file = 'Apprentice_Chef_Dataset.xlsx'
original_df = pd.read_excel(file)



################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# Calculating average price per meal

chef_data = original_df.copy()

# creating a column for avg price meal

chef_data['AVG_PRICE_MEAL'] = chef_data['REVENUE']/chef_data['TOTAL_MEALS_ORDERED'].round(2)

chef_data = pd.DataFrame(chef_data)

# creating ordered beverage column
chef_data['ORDERED_BEVERAGES'] = 0

# looping to find Miss.
for index, val in chef_data.iterrows():
    
    if chef_data.loc[index, 'AVG_PRICE_MEAL'] > 23:
        chef_data.loc[index, 'ORDERED_BEVERAGES'] = 1
        
    elif chef_data.loc[index,'AVG_PRICE_MEAL'] <= 23:
        chef_data.loc[index, 'ORDERED_BEVERAGES'] = 0
    
    else:
        print('error')
        
chef_data.to_excel('Apprentice_Chef_featured.xlsx',
                 index = False)

#Creating a column for % of unique meals purchased
chef_data['PER_UNIQUE_MEALS']= round(chef_data['UNIQUE_MEALS_PURCH']/ chef_data['TOTAL_MEALS_ORDERED']*100, 2)


#Based on the previous column creating a "williness to try new things"

chef_data['WILLIGNESS_NEW_PRODUCTS'] = 0
chef_data['WILLIGNESS_NEW_PRODUCTS'][chef_data['PER_UNIQUE_MEALS'] >= 20]= 1


# 14 % of customers show willingness to try new products ----> round(267/1946, 2)
# print(chef_data['WILLIGNESS_NEW_PRODUCTS'].sum()) ------> 267 customers the most willing to try new things


chef_data.to_excel('Apprentice_Chef_featured.xlsx',
                 index = False)
# Follow recomendations appears to be significant. Create a column based on this 

chef_data = pd.read_excel('Apprentice_Chef_featured.xlsx')

chef_data['FOLLOWED_RECOMMENDATIONS'] = 0
chef_data['FOLLOWED_RECOMMENDATIONS'][chef_data['FOLLOWED_RECOMMENDATIONS_PCT'] > 30] = 1


# Splitting personal emails (creating 5 new features)

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in chef_data.iterrows():
    
    # splitting email domain at '@'
    class_email = chef_data.loc[index,'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(class_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


#concatenating with original DataFrame
# renaming column to concatenate
email_df.columns = ['0' , 'EMAIL_DOMAIN']

# creating domain email column
chef_data = pd.concat([chef_data, email_df['EMAIL_DOMAIN']],
                     axis = 1)



# 3 email domain class
PERSONAL_EMAIL= ['@gmail.com', '@yahoo.com', '@protonmail.com']
JUNK_EMAIL = ['@me.com', '@aol.com', '@hotmail.com', '@ive.com', '@msn.com', '@passport.com']

# created domain class

# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in chef_data['EMAIL_DOMAIN']:
    
    if '@' + domain in PERSONAL_EMAIL:
        placeholder_lst.append('PERSONAL_@')
        

    elif '@' + domain in JUNK_EMAIL:
        placeholder_lst.append('JUNK_@')


    else:
        placeholder_lst.append('PROFESSIONAL_@')


# concatenating with original_2
chef_data['DOMAIN_CLASS'] = pd.Series(placeholder_lst)

# Creating variable 1/0 for emails
EMAIL_CLASS = pd.get_dummies(chef_data['DOMAIN_CLASS'])

# joining codings together
chef_data= chef_data.join([EMAIL_CLASS])


# saving new columns
email_dummies_columns = chef_data.columns

#the professional email registrated are rom forbe 500 companies, it is unlikly that these companies 
#hired people with no degrees, this means that these group of people are probably over 21



# Based on the previous column creating a "williness to try new things"

chef_data['OVER_21'] = 1
chef_data['OVER_21'][chef_data['EMAIL_DOMAIN'] == 'gmail.com' ] = 0

#setting outlier thresholds

REVENUE_hi = 5000 

TOTAL_MEALS_ORDERED_hi = 220   

UNIQUE_MEALS_PURCH_hi = 9   

CONTACTS_W_CUSTOMER_SERVICE_hi = 12
CONTACTS_W_CUSTOMER_SERVICE_lo = 2.5

AVG_TIME_PER_SITE_VISIT_hi = 200    

CANCELLATIONS_BEFORE_NOON_hi = 6

MOBILE_LOGINS_hi = 6
MOBILE_LOGINS_lo = 5

PC_LOGINS_hi = 2
PC_LOGINS_lo = 1

EARLY_DELIVERIES_hi = 4

LATE_DELIVERIES_hi = 10  

AVG_PREP_VID_TIME_hi = 300

LARGEST_ORDER_SIZE_hi = 8   
LARGEST_ORDER_SIZE_lo = 2

MASTER_CLASSES_ATTENDED_hi = 1
MASTER_CLASSES_ATTENDED_lo = 0

MEDIAN_MEAL_RATING_hi = 4
MEDIAN_MEAL_RATING_lo = 2

AVG_CLICKS_PER_VISIT_hi = 17.5
AVG_CLICKS_PER_VISIT_lo = 10

TOTAL_PHOTOS_VIEWED_hi = 350

AVG_PRICE_MEAL_hi = 75

PER_UNIQUE_MEALS_hi = 25

# renaming my data frame to reuse an old code

original_2 = chef_data.copy()

# developing features (columns) for outliers
#Explanatory variables 

original_2['OUT_REVENUE'] = 0
condition_hi_re = original_2.loc[0:,'OUT_REVENUE'][original_2['REVENUE'] > REVENUE_hi]

original_2['OUT_REVENUE'].replace(to_replace = condition_hi_re,
                                value      = 1,
                                inplace    = True)


#TOTAL_MEALS_ORDERED
original_2['OUT_TOTAL_MEALS_ORDERED'] = 0
condition_hi_meals = original_2.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][original_2['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]

original_2['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi_meals,
                                value      = 1,
                                inplace    = True)


#UNIQUE_MEALS_PURCH_
original_2['OUT_UNIQUE_MEALS_PURCH'] = 0
condition_hi_unq = original_2.loc[0:,'OUT_UNIQUE_MEALS_PURCH'][original_2['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]

original_2['OUT_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi_unq,
                                value      = 1,
                                inplace    = True)

#CONTACTS_W_CUSTOMER_SERVICE high
original_2['OUT_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi_cs = original_2.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][original_2['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]

original_2['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi_cs,
                                value      = 1,
                                inplace    = True)

condition_lo_cs = original_2.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][original_2['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]

original_2['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo_cs,
                                 value      = 1,
                                 inplace    = True)

#AVG_TIME_PER_SITE_VISIT
original_2['OUT_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi_sv = original_2.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][original_2['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

original_2['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi_sv,
                                value      = 1,
                                inplace    = True)

#CANCELLATIONS_BEFORE_NOON
original_2['OUT_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi_bnoon = original_2.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][original_2['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

original_2['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi_bnoon,
                                value      = 1,
                                inplace    = True)


# MOBILE_LOGINGS
original_2['OUT_MOBILE_LOGINS'] = 0
condition_hi_mlg = original_2.loc[0:,'OUT_MOBILE_LOGINS'][original_2['MOBILE_LOGINS'] > MOBILE_LOGINS_hi]
condition_lo_mlg = original_2.loc[0:,'OUT_MOBILE_LOGINS'][original_2['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]

original_2['OUT_MOBILE_LOGINS'].replace(to_replace = condition_hi_mlg,
                                 value      = 1,
                                 inplace    = True)

original_2['OUT_MOBILE_LOGINS'].replace(to_replace = condition_lo_mlg,
                                 value      = 1,
                                 inplace    = True)


# PC_lOGINS
original_2['OUT_PC_LOGINS'] = 0
condition_hi_pc = original_2.loc[0:,'OUT_PC_LOGINS'][original_2['PC_LOGINS'] > PC_LOGINS_hi]
condition_lo_pc = original_2.loc[0:,'OUT_PC_LOGINS'][original_2['PC_LOGINS'] < PC_LOGINS_lo]

original_2['OUT_PC_LOGINS'].replace(to_replace = condition_hi_pc,
                                 value      = 1,
                                 inplace    = True)

original_2['OUT_PC_LOGINS'].replace(to_replace = condition_lo_pc,
                                 value      = 1,
                                 inplace    = True)


#EARLY_DELIVERIES
original_2['OUT_EARLY_DELIVERIES'] = 0
condition_hi_ed = original_2.loc[0:,'OUT_EARLY_DELIVERIES'][original_2['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

original_2['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_hi_ed,
                                value      = 1,
                                inplace    = True)


#LATE_DELIVERIES
original_2['OUT_LATE_DELIVERIES'] = 0
condition_hi_ld = original_2.loc[0:,'OUT_LATE_DELIVERIES'][original_2['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]

original_2['OUT_LATE_DELIVERIES'].replace(to_replace = condition_hi_ld,
                                value      = 1,
                                inplace    = True)



#AVG_PREP_VID_TIME
original_2['OUT_AVG_PREP_VID_TIME'] = 0
condition_hi_vid= original_2.loc[0:,'OUT_AVG_PREP_VID_TIME'][original_2['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]

original_2['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi_vid,
                                value      = 1,
                                inplace    = True)


#LARGEST_ORDER_SIZE
original_2['OUT_LARGEST_ORDER_SIZE'] = 0
condition_hi_large= original_2.loc[0:,'OUT_LARGEST_ORDER_SIZE'][original_2['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]

original_2['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi_large,
                                value      = 1,
                                inplace    = True)

condition_lo_large= original_2.loc[0:,'OUT_LARGEST_ORDER_SIZE'][original_2['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]
original_2['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo_large,
                                value      = 1,
                                inplace    = True)



#MASTER CLASS ATTENDED
original_2['OUT_MASTER_CLASSES_ATTENDED'] = 0
condition_hi_mas= original_2.loc[0:,'OUT_MASTER_CLASSES_ATTENDED'][original_2['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]

original_2['OUT_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi_mas,
                                value      = 1,
                                inplace    = True)

condition_lo_mas= original_2.loc[0:,'OUT_MASTER_CLASSES_ATTENDED'][original_2['MASTER_CLASSES_ATTENDED'] < MASTER_CLASSES_ATTENDED_lo]

original_2['OUT_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_lo_mas,
                                value      = 1,
                                inplace    = True)


#MEDIAN_MEAL_RATING
original_2['OUT_MEDIAN_MEAL_RATING'] = 0
condition_hi_med= original_2.loc[0:,'OUT_MEDIAN_MEAL_RATING'][original_2['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]

original_2['OUT_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi_med,
                                value      = 1,
                                inplace    = True)

condition_lo_med= original_2.loc[0:,'OUT_MEDIAN_MEAL_RATING'][original_2['MEDIAN_MEAL_RATING'] <MEDIAN_MEAL_RATING_lo]

original_2['OUT_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo_med,
                                value      = 1,
                                inplace    = True)

#AVG_CLICKS_PER_VISIT low/high
original_2['OUT_AVG_CLICKS_PER_VISIT'] = 0
condition_hi_cl = original_2.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][original_2['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo_cl = original_2.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][original_2['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

original_2['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi_cl,
                                 value      = 1,
                                 inplace    = True)

original_2['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo_cl,
                                 value      = 1,
                                 inplace    = True)


#TOTAL_PHOTOS_VIEWED
original_2['OUT_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi_ph= original_2.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][original_2['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]

original_2['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi_ph,
                                value      = 1,
                                inplace    = True)


# AVG_PRICE_PER_MEAL

original_2['OUT_AVG_PRICE_MEAL'] = 0
condition_hi_mealp= original_2.loc[0:,'OUT_AVG_PRICE_MEAL'][original_2['AVG_PRICE_MEAL'] > AVG_PRICE_MEAL_hi]

original_2['OUT_AVG_PRICE_MEAL'].replace(to_replace = condition_hi_mealp,
                                value      = 1,
                                inplace    = True)


#PER_UNIQUE_MEAL

original_2['OUT_PER_UNIQUE_MEALS'] = 0
condition_hi_ump= original_2.loc[0:,'OUT_PER_UNIQUE_MEALS'][original_2['PER_UNIQUE_MEALS'] > PER_UNIQUE_MEALS_hi]

original_2['OUT_PER_UNIQUE_MEALS'].replace(to_replace = condition_hi_ump,
                                value      = 1,
                                inplace    = True)
# CHECKPOINT
original_2.to_excel('Apprentice_Chef_featured.xlsx', index= False)
chef_data = pd.read_excel('Apprentice_Chef_featured.xlsx')

# droping categorical features
chef_model = chef_data.drop(['NAME','EMAIL','FIRST_NAME','FAMILY_NAME', 'EMAIL_DOMAIN','DOMAIN_CLASS'], axis=1)


    
chef_full = chef_model.copy()


################################################################################
# Train/Test Split
################################################################################
#train test split based on full dataset
#declaring explanatory variables
chef_x= chef_full.drop('CROSS_SELL_SUCCESS', axis = 1)


# declaring response variable
chef_y = chef_model.loc[ : , 'CROSS_SELL_SUCCESS']

X_train, X_test,y_train, y_test = train_test_split(
            chef_x,
            chef_y,
            test_size    = 0.25,
            random_state = 222,
            stratify     = chef_y)





################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model
#INSTANTIATING a classification tree object
tree_pruned      = DecisionTreeClassifier(max_depth = 4,
                                          min_samples_leaf = 25,
                                          random_state = 222)


# FITTING the training data
tree_pruned_fit  = tree_pruned.fit(X_train, y_train)


# PREDICTING on new data
tree_pred = tree_pruned_fit.predict(X_test)




################################################################################
# Final Model Score (score)
################################################################################



train_score = tree_pruned_fit.score(X_train, y_train).round(4)
test_score= tree_pruned_fit.score(X_test, y_test).round(4)
auc_score= roc_auc_score(y_true  = y_test,
                                          y_score = tree_pred).round(4)

# SCORING the model
print(f'Training ACCURACY: {train_score}')
print(f'Testing  ACCURACY: {test_score}')
print(f'AUC Score        : {auc_score}')


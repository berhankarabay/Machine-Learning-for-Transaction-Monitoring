#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 01:47:35 2023

@author: berhankarabay
"""

#%% Load libraries

import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier



pd.set_option('display.max_columns', 700)
pd.set_option('display.max_rows', 400)
pd.set_option('display.min_rows', 10)
pd.set_option('display.expand_frame_repr', True)
pd.options.display.float_format = "{:,.2f}".format


#%% load dataset

filename = 'HI-Small_Trans.csv'
df = pd.read_csv(filename)
print(df.columns)

#%% 3. Data Description & Visualizations

df.info()
df.head()
print(df.describe())

#Counting the occurrences of fraud and no fraud

occ = df['Is Laundering'].value_counts()
occ

# Print the ratio of fraud cases
ratio_cases = occ/len(df.index)
print(f'Ratio of fraudulent cases: {ratio_cases[1]}\nRatio of non-fraudulent cases: {ratio_cases[0]}')



#%% Feature Engineering | Timestamp

#Due to the patterns given in our data, we choose to use only dates.

# Convert the 'Time Stamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')

# Extract the day and create a new column 'Day'
df['Day'] = df['Timestamp'].dt.day
df = df.drop(['Timestamp'], axis=1)
# Reorder the columns

print(df)

#%%Removing alphabetical values from Account IDs


import hashlib

unique_count = df['Account'].nunique()
unique_count1 = df['Account.1'].nunique()


print("Number of unique Account values before:", unique_count)
print("Number of unique Account.1 values before:", unique_count1)


# Create a hash function to generate unique integer IDs

def generate_unique_id(account_number):
    hash_str = hashlib.sha256(account_number.encode()).hexdigest()
    return int(hash_str, 16) % 10**8

# Create a dictionary to store mappings
account_id_mapping = {}

# Function to get or generate unique ID for an account number
def get_or_generate_id(account_number):
    if account_number not in account_id_mapping:
        account_id_mapping[account_number] = generate_unique_id(account_number)
    return account_id_mapping[account_number]

# Apply the mappings to your columns
df['Account_ID'] = [get_or_generate_id(account) for account in df['Account']]
df['Account.1_ID'] = [get_or_generate_id(account) for account in df['Account.1']]

unique_count = df['Account_ID'].nunique()
unique_count1 = df['Account.1_ID'].nunique()


print("Number of unique Account_ID values after:", unique_count)
print("Number of unique Account.1_ID values after:", unique_count1)

df = df.drop(['Account', 'Account.1'], axis=1)

# Print the result
print(df)


#%% Feature Engineering


#Checking if the amount paid and received is equal. Looks like its not.

different_attributes = df[df['Amount Received'] != df['Amount Paid']]
print(different_attributes)


# To simplfy and reduce dimensions, we'll create a new column "currencychange"(1 for yes, 0 for no) and remove currency columns"
# We'll convert all "Receiving Currency" to USD. To apply exchange rates we'll use Open Exchange Rates API.

#First mapping of non-standard currency names to ISO currency codes
currency_mapping = {
    'US Dollar': 'USD',
    'Australian Dollar': 'AUD',
    'Bitcoin': 'BTC',
    'Brazil Real': 'BRL',
    'Canadian Dollar': 'CAD',
    'Euro': 'EUR',
    'Mexican Peso': 'MXN',
    'Ruble': 'RUB',
    'Rupee': 'INR',
    'Swiss Franc': 'CHF',
    'Shekel': 'ILS',
    'Saudi Riyal': 'SAR',
    'UK Pound': 'GBP',
    'Yen': 'JPY',
    'Yuan': 'CNY',
    
}

# Update 'currency' column using the mapping
df['Receiving Currency'] = df['Receiving Currency'].map(currency_mapping)
df['Payment Currency'] = df['Payment Currency'].map(currency_mapping)



#%% Feature Engineering | Functions


# Define a function to check if the currencies are the same
def check_currencies(data):
    if data['Receiving Currency'] != data['Payment Currency']:
        return 1
    else:
        return 0

#Surge indicator
def surge_indicator(data):
    '''Creates a new column which has 1 if the transaction amount is greater than the threshold
    else it will be 0'''
    data['surge']=[1 if n>10000 else 0 for n in data['Amount Received']]

#Frequency indicator
def frequency_receiver(data):
    '''Creates a new column which has 1 if the receiver receives money from many individuals
    else it will be 0'''
    data['freq_Dest']=data['Account.1_ID'].map(data['Account.1_ID'].value_counts())
    data['freq_dest']=[1 if n>30 else 0 for n in data['freq_Dest']]
    data.drop(['freq_Dest'],axis=1,inplace = True)



#%% Feature Engineering | Applying Functions 

# Applying check_currencies function
df['currency_change'] = df.apply(check_currencies, axis=1)
df['currency_change'].value_counts()

# Applying surge_indicator function
surge_indicator(df)
df['surge'].value_counts()
    
# Applying frequency_receiver function
frequency_receiver(df)
df['freq_dest'].value_counts()
    
#%% Feature Engineering | Fixing currencies to USD


# Get exchange rate data (example: using Open Exchange Rates API)
api_key = 'b15a7a606acf4ccc8c98330f30d5268b'
base_currency = 'USD'
exchange_url = f'https://openexchangerates.org/api/latest.json?app_id={api_key}&base={base_currency}'
response = requests.get(exchange_url)
exchange_rates = response.json()['rates']

# Merge exchange rates with the original DataFrame
df = df.merge(pd.DataFrame(exchange_rates.items(), columns=['Receiving Currency', 'exchange_rate']), on='Receiving Currency')

# Convert to USD
df['amount_usd'] = (df['Amount Received'] / df['exchange_rate']).astype(int)

print(df)




#%% Feature Engineering | Dropping unnecessary columns


columns_to_drop = ['Amount Paid','Amount Received','Receiving Currency','Payment Currency','exchange_rate']
df = df.drop(columns_to_drop, axis=1)

print(df.columns)
# Reorder the columns
new_column_order = ['Day'] + [col for col in df.columns if col != 'Day' and col != 'Is Laundering'] + ['Is Laundering']
df = df[new_column_order]


print(df)



#%% Feature Engineering

df['Payment Format'] = pd.factorize(df['Payment Format'])[0] + 1

#correlation matrix
correlations = df.corr()
k = 10 #number of variables for heatmap
cols = correlations.nlargest(k, 'Is Laundering')['Is Laundering'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



#%% Splitting the dataset

array = df.values
X = df.drop('Is Laundering',axis=1)
Y = df['Is Laundering']

test_size = 0.20
seed = 7
X_model, X_test, Y_model, Y_test = train_test_split(X,Y,test_size=test_size, random_state= seed)

validation_size = 0.25
X_train, X_val, Y_train, Y_val = train_test_split(X_model, Y_model, test_size=validation_size, random_state=seed)

#%%Standardizing the numerical columns

method = SMOTE()

X_resampled, Y_resampled = method.fit_resample(X_train, Y_train)


#%% Evaluate algorithms: baseline 


# spot-check algorithms
models = []
models.append(('LR',LogisticRegression(solver='liblinear')))
models.append(('ADA',AdaBoostClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('NB',GaussianNB()))
models.append(('XGBC',XGBClassifier()))

# Compare algorithms
results = []
names = []
for name, model in models:
    model.fit(X_resampled, Y_resampled)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(Y_val, predictions)
    f1 = f1_score(Y_val, predictions, average='weighted')
    print("%s Validation Accuracy: %.2f%%" % (name, accuracy * 100))
    print(f"F1 Score: {f1:.4f}")
    names.append(name)
    results.append(accuracy)
    



#XGBC was best performing amonst classification models.No need to scale our features when using XGBClassifier.

#%% Tuning XGBC (Kept simple due to computing time constraints)


cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
param_grid = {

  "max_depth": list(range(1, 5)),

  "n_estimators": list(range(1, 5)),

  "learning_rate": [0.01, 0.1, 1],
  
}

model = GridSearchCV(XGBClassifier(), param_grid, cv=cv, scoring="roc_auc")
model.fit(X_resampled, Y_resampled)

best_params = model.best_params_
print("Best Parameters:", best_params)

#%% Tuning Concrete XGBC (Kept simple due to computing time constraints)

time_begin = time.time()

# The Concrete ML model needs an additional parameter used for quantization
param_grid["n_bits"] = [3]

# Instantiate and fit the model through grid-search cross-validation
X_resampled = X_resampled.astype(np.float32)

concrete_model = GridSearchCV(ConcreteXGBClassifier(), param_grid, cv=cv, scoring="roc_auc")
concrete_model.fit(X_resampled, Y_resampled)

cv_concrete_duration = time.time() - time_begin

print(f"Best hyper-parameters found in {cv_concrete_duration:.2f}s :", concrete_model.best_params_)



#%%  Predicting Outcomes

# Compute the predictions in clear using XGBoost
clear_predictions = model.predict(X_test)

print("Accuracy Score:",accuracy_score(Y_test,clear_predictions))
print("Weighted F1 Score",f1_score(Y_test,clear_predictions, average='weighted'))
print(confusion_matrix(Y_test,clear_predictions))
print(classification_report(Y_test,clear_predictions))

f1_minority = f1_score(Y_test, clear_predictions, pos_label=1)
print("F1 Score (Minority Class):", f1_minority)

# Compute the predictions in clear using Concrete ML
clear_quantized_predictions = concrete_model.predict(X_test)

print(accuracy_score(Y_test,clear_quantized_predictions))
print(f1_score(Y_test,clear_quantized_predictions, average='weighted'))
print(confusion_matrix(Y_test,clear_quantized_predictions))
print(classification_report(Y_test,clear_quantized_predictions))

f1_minority = f1_score(Y_test, clear_quantized_predictions, pos_label=1)
print("F1 Score (Minority Class):", f1_minority)

# Compile the Concrete ML model on a subset
fhe_circuit = concrete_model.best_estimator_.compile(X_resampled.head(100))



# Generate the keys
# This step is not absolutely necessary, as keygen() is called, when necessary,
# within the predict method.
# However, it is useful to run it beforehand in order to be able to
# measure the prediction executing time separately from the key generation one
time_begin = time.time()
fhe_circuit.keygen()
key_generation_duration = time.time() - time_begin


# Compute the predictions in FHE using Concrete ML
time_begin = time.time()
fhe_predictions = concrete_model.best_estimator_.predict(X_test, fhe="execute")
prediction_duration = time.time() - time_begin

print(f"Key generation time: {key_generation_duration:.2f}s")
print(f"Total execution time for {len(clear_predictions)} inferences: {prediction_duration:.2f}s")
print(f"Execution time per inference in FHE: {prediction_duration / len(clear_predictions):.2f}s")


number_of_equal_preds = np.sum(fhe_predictions == clear_quantized_predictions)
pred_similarity = number_of_equal_preds / len(clear_predictions) * 100
print(

  "Prediction similarity between both Concrete-ML models" 

  f"(quantized clear and FHE): {pred_similarity:.2f}%"
)

accuracy_fhe = np.mean(fhe_predictions == Y_test) * 100
print(
    "Accuracy of prediction in FHE on the test set " f"{accuracy_fhe:.2f}%",
)




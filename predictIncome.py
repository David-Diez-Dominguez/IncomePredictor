import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Get the current directory
current_dir = os.path.dirname(__file__)

# Path to the Excel file
excel_file_path = os.path.join(current_dir, 'income.xlsx')

# Import excel file and read save it as a dataFrame
df = pd.read_excel(excel_file_path, header=0)
#x are all the variables except the target variable
x = df.iloc[:,:-6].values
#y is the target variable (income)
y = df.iloc[:,-6].values

# For the predictor to perform, Integer values are required
# For each colum that contains Values of type String, we change each String values of a row into a distinct Integer number

#The fit_transfor6m method of LabelEncoder is used to fit the encoder on the data in the
#specified column and transform the string values into numerical labels.
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

string_columns_indices = [1, 2, 4, 5, 6, 7, 8, 12]

for col_index in string_columns_indices:
     x[:, col_index] = label_encoder.fit_transform(x[:, col_index])

y=label_encoder.fit_transform(y)


# Before training the data, we need to split into test and train data
# training data, to train the model (80% of the data)
# test data is needed fot the prediction to have unexpected data (20% of the data)
# We do both for x and y
from sklearn.model_selection import train_test_split
# The random_state attribute can be a random number.For the same random_state the set of values for the training and testing data is always the same.
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# #Here the model is trained
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train,Y_train)

# Test the accuracy of the model
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(accuracy_score(Y_test,y_pred))  

# Here we can give the modal our own test cases to cpredict if a Person makes more or less than 50K a year depending on the attributes
test_data = {
    'Age': 39,
    'Employed': 'State-gov',
    'Education': 'HS-grad',
    'EN': 13,
    'Marital Status': 'Never-married',
    'Occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'Race': 'White',
    'Sex': 'Male',
    'capital gain': 0,
    'Capital loss': 0,
    'hours/week': 40,
    'native country': 'United-States'
    # 'Age': 52,
    # 'Employed': 'Self-emp-not-inc',
    # 'Education': 'HS-grad',
    # 'EN': 9,
    # 'Marital Status': 'Married-civ-spouse',
    # 'Occupation': 'Exec-managerial',
    # 'relationship': 'Husband',
    # 'Race': 'White',
    # 'Sex': 'Male',
    # 'capital gain': 1200,
    # 'Capital loss': 0,
    # 'hours/week': 45,
    # 'native country': 'United-States'
}

#Convert the dicctionary into a dataFrame
test_data_df = pd.DataFrame([test_data])

#here we need to labelencode the test case attributes again
le1=LabelEncoder()
le2=LabelEncoder()
le4=LabelEncoder()
le5=LabelEncoder()
le6=LabelEncoder()
le7=LabelEncoder()
le8=LabelEncoder()
le12=LabelEncoder()
test_data_df['Employed']=le1.fit_transform(test_data_df['Employed'])
test_data_df['Education']=le2.fit_transform(test_data_df['Education'])
test_data_df['Marital Status']=le2.fit_transform(test_data_df['Marital Status'])
test_data_df['Occupation']=le5.fit_transform(test_data_df['Occupation'])
test_data_df['relationship']=le6.fit_transform(test_data_df['relationship'])
test_data_df['Race']=le7.fit_transform(test_data_df['Race'])
test_data_df['Sex']=le8.fit_transform(test_data_df['Sex'])
test_data_df['native country']=le12.fit_transform(test_data_df['native country'])

# print the Result
result = model.predict(test_data_df.values)
print(int(result))
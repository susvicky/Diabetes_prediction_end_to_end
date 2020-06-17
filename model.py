import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import pickle

data = pd.read_csv("datasets_228_482_diabetes.csv")
data.head()

from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'SkinThickness']
predicted_class = ['Outcome']


X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

model = RandomForestClassifier(random_state=10)
model.fit(X_train, y_train.ravel())


print(model.score(X_test,y_test))

from sklearn.externals import joblib 

# Save the model as a pickle in a file 
pickle.dump(classifier,open( 'final_model.pkl','wb')) 

# Load the model from the file 
my_model = joblib.load('final_model.pkl','rb') 

print(my_model.score(X_test,y_test))

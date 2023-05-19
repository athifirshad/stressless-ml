import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#load dataset
hrvds = pd.read_csv('train.csv') #train dataset here
hrvds.shape
# seperating data and label
data = hrvds.drop(columns = 'condition', axis = 1)
label = hrvds['condition']
#train test split
data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.2,stratify=label,random_state=2)
#standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

data_train_std = scaler.fit_transform(data_train)

data_test_std = scaler.transform(data_test)

# importing tensorflow and Keras
import tensorflow as tf 
tf.random.set_seed(3)
from tensorflow import keras
# setting up the layers of Neural Network
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(34,)),
                          keras.layers.Dense(25, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])
# compiling the Neural Network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# training the Neural Network
history = model.fit(data_train_std, label_train, validation_split=0.1, epochs=10)
#accuracy on data
loss, accuracy = model.evaluate(data_test_std, label_test)
print('Accuracy score of the test data : ', accuracy)

loss, accuracy = model.evaluate(data_train_std, label_train)
print('Accuracy score of the train data : ', accuracy)
import csv
# Open the CSV file
with open('TEST1.csv', newline='') as csvfile:
    
    # Create a CSV reader object
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    arr = []
    # Loop over each row in the CSV file

    for row in reader:
        arr.append( row[0])
    
    
#input_data = (768.5960196,786.18077,59.55781547,7.59287746,7.592163954,7.843905791,78.57847301,0.266666667,0,5.370262196,84.05609467,0.208785088,-0.960595519,0.000155237,-0.00098501,0.010097214,0.006146008,0.006145997,1.642889908,0.208785088,-0.960595519,299.0137274,53.52424654,241.8551493,43.29271016,93.1511744,17.78210258,3.183043303,6.848825604,558.6509793,13.60104342,0.073523771,1.961080905,1.229617876)
# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(arr)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
#print(prediction_label)

if(prediction_label[0] == 0):
  print('no stress')

else:
  print('stress')

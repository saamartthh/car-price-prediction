import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pk

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

def clean_data(value):
    value = value.split(' ')[0]
    value = value.strip()
    if value == '':
        value = 0
    return float(value)

# Load data
cars_data = pd.read_csv('Cardetails.csv')

# Data cleaning and preprocessing
cars_data.drop(columns=['torque'], inplace=True)
cars_data.dropna(inplace=True)
cars_data.drop_duplicates(inplace=True)

# Feature engineering
cars_data['name'] = cars_data['name'].apply(get_brand_name)
cars_data['mileage'] = cars_data['mileage'].apply(clean_data)
cars_data['max_power'] = cars_data['max_power'].apply(clean_data)
cars_data['engine'] = cars_data['engine'].apply(clean_data)

# Apply encodings
cars_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
    'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
    'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
    'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    inplace=True)

cars_data['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3], inplace=True)
cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4], inplace=True)
cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
    'Fourth & Above Owner', 'Test Drive Car'],
    [1,2,3,4,5], inplace=True)

# Prepare training data
input_data = cars_data.drop(columns=['selling_price'])
output_data = cars_data['selling_price']

# Train model
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)

# Save model
pk.dump(model, open('model.pkl','wb'))
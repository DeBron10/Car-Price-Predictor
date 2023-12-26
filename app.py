from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

app = Flask(__name__)
cors = CORS(app)

# Load your machine learning model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Azure Blob Storage credentials
account_name = 'mycsvstore'
account_key = 'xfpG78DhTaaLputWhMaSoWFRdd9IXOAzw/iTNKIvRIMDATRa3kROdK3By6Bdzm5Mwz50tSYTBg8K+AStc+WHrw=='
container_name = 'csv-store'
blob_name = 'Cleaned_Car_data.csv'

# Create a BlobServiceClient
blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

# Download the CSV file from Azure Blob Storage
with open('Cleaned_Car_data.csv', 'wb') as file:
    data = blob_client.download_blob()
    file.write(data.readall())

# Read the CSV file
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Use the loaded model for prediction
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run()

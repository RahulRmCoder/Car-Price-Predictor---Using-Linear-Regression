from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv("Cleaned Car.csv")
@app.route('/',methods=['Get','POST'])
def index():
    companies= sorted(car['company'].unique())
    car_models =sorted(car['name'].unique())
    year =sorted(car['year'].unique(),reverse=True)
    fuel_type=sorted(car['fuel_type'].unique())
    companies.insert(0,'Select Company')
    year.insert(0,"Car Purchase Year")
    fuel_type.insert(0,"Car Fuel Type")

    return render_template('index.html',companies=companies,car_models=car_models,years=year,fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],data=np.array([car_model,company,year,kms_driven,fuel_type]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0],2))
if __name__=="__main__":
    app.run(debug=True)
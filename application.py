from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open("LinearRegressionModel.pkl",'rb'))
car=pd.read_csv('Clean.csv')

@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    car_model=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    companies.insert(0,"Select Company")
    return render_template('index.html', companies=companies,car_model=car_model,year=year,fuel_type=fuel_type)


@app.route('/predictor',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    fuel_type=request.form.get('fuel_type')
    year=int(request.form.get('year'))
    kms_driven=int(request.form.get('kilo_driven'))
    print(company,car_model,fuel_type,year,kms_driven) 
    

    Prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(Prediction[0])
    return str(np.round(Prediction[0],2))

if __name__== "__main__":
    app.run(debug=True)
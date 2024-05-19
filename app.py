from flask import Flask, render_template, request
import pickle
import numpy as np

model_data = pickle.load(open("models/exp_predictor.pkl", "rb"))

model = model_data['model']
encoder = model_data['encoder']
scaler = model_data['scaler']

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get('name')
    age = float(request.form.get('age', 18))
    bmi = float(request.form.get('bmi', 28))
    children = float(request.form.get('children', 0))

    smoker = request.form['Smoker']
    sex = request.form['Gender']
    region = request.form['Region']
 
    numeric_data = np.array([age, bmi, children])
    scaled_numeric_data = scaler.transform(numeric_data.reshape(1, -1))
    if(sex=='male'):
        sex=1.0
    else:
        sex=0.0

    if(smoker=='yes'):
        smoker=1.0
    else:
        smoker=0.0
    categorical_data = np.array([sex, smoker]).reshape(1,-1)
    ohe_region = encoder.transform(np.array([[region]])).toarray().reshape(1, -1)

    input_data = np.concatenate((scaled_numeric_data, categorical_data, ohe_region), axis=1)
    prediction = model.predict(input_data)
    title = "Your Predicted Annual Medical Expenditutre is :"
    prediction = abs(prediction)
    return render_template("index.html", prediction=prediction, input_data=input_data, title= title)

if __name__ == "__main__":
    app.run(debug=True)
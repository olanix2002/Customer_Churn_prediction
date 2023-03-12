#import pickle  # To read our pickled model and scaler
from flask import Flask, request, jsonify, url_for, render_template  # Importing falsk framework and neccesary libraries
import numpy as np #Importing numpy for data conversion and reshaping
#from tensorflow.keras.models import load_model
import joblib as jb


app=Flask(__name__, static_folder='/static')  #initializing flask


model = jb.load(open('model.h5', 'rb'))

@app.route('/')  # our home page
def home():
    return render_template('index.html') # Returns an html file that collect the input of our data and have a button for prediction


@app.route('/predict', methods=['POST'])  # A web page links to our form and prediction button in our nhtml file
def predict():
    data=[float(x) for x in request.form.values()]  # Turning our form values to float and saving the in a variable
    final_input=np.array(data).reshape(1,-1)  # reshaping the size to fit into the model
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("index.html", prediction_text="The house price prediction is {}".format(output))     # return our predicted value and the basic html page



@app.route('/predict_api', methods=['POST'])  # A web page links to our form and prediction button in our nhtml file
def predict():
    data = request.get_json()['data']
    # Convert the input data to a numpy array
    features = np.array([data['CreditScore'],
    0 if data['Geography'] == 'France' else 1 if data['Geography'] == 'Spain' else 2, # convert Geography to numeric
    0 if data['Gender'] == 'Male' else 1, # convert Gender to numeric
    data['Age'],
    data['Tenure'],
    data['Balance'],
    data['NumOfProducts'],
    1 if data['HasCrCard'] == 'Yes' else 0,
    1 if data['IsActiveMember'] == 'Yes' else 0,
    data['EstimatedSalary']
])

    # Make the prediction using the model
    prediction = model.predict(features.reshape(1, -1))

    # Convert the prediction to a float
    prediction = float(prediction[0][0])
    print(prediction)
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction}) 


if __name__=="__main__":   # To start the app
    app.run(debug=True)    # Enabling debugging
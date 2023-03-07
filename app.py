from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the saved model from a file
model = load_model('model.h5')

# Set Flask app configuration to production
app.config['ENV'] = 'production'

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
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
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = data = request.form['data']

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

    prediction = model.predict(features.reshape(1, -1))

    # Convert the prediction to a float
    prediction = float(prediction[0][0])
    print(prediction)
    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    # Set Flask app to listen to all IP addresses and use the correct port
    app.run(host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model from a file
model = load_model('model.h5')

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()['data']

    # Convert the input data to a numpy array
    features = np.array([
        data['CreditScore'],
        data['Geography'],
        data['Gender'],
        data['Age'],
        data['Tenure'],
        data['Balance'],
        data['NumOfProducts'],
        data['HasCrCard'],
        data['IsActiveMember'],
        data['EstimatedSalary']
    ])

    # Make the prediction using the model
    prediction = model.predict(features.reshape(1, -1))

    # Convert the prediction to a float
    prediction = float(prediction[0][0])

    # Return the prediction as a JSON response
    return {'prediction': prediction}

if __name__ == "__main__":
    app.run(debug=True)




'''from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model from a file
model = load_model('model.h5')

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()['data']

    # Convert the input data to a numpy array
    features = np.array([
        data['CreditScore'],
        data['Geography'],
        data['Gender'],
        data['Age'],
        data['Tenure'],
        data['Balance'],
        data['NumOfProducts'],
        data['HasCrCard'],
        data['IsActiveMember'],
        data['EstimatedSalary']
    ])

    # Make the prediction using the model
    prediction = model.predict(features.reshape(1, -1))

    # Convert the prediction to a float
    prediction = float(prediction[0][0])
    

    # Return the prediction as a JSON response
    return {'prediction': prediction}
if __name__ == "__main__":
    app.run(debug=True)'''

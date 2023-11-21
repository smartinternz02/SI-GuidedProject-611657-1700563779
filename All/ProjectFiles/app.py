from flask import Flask, render_template, request
import numpy as np
import pickle
# from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model
with open("Travel.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the MinMaxScaler
with open("MinMaxScaler.pkl", "rb") as scaler_file:
    ms = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Age = int(request.form['Age'])
    
    EmployementType = request.form['EmploymentType']
    if EmployementType == 'PrivateSector/SelfEmployed':
        EmployementType = 1
    else:
       EmployementType =  0
    
    GraduateOrNot = request.form['GraduateOrNot']
    if GraduateOrNot == 'Yes':
        GraduateOrNot = 1
    else :
        GraduateOrNot = 0
    
    AnnualIncome = float(request.form['AnnualIncome'])
    
    FamilyMembers = int(request.form['FamilyMembers'])
    
    chronicDiseases = request.form['ChronicDiseases']
    if chronicDiseases == 'Yes':
        chronicDiseases = 1
    else:
        chronicDiseases = 0
    
    FrequentFlyer = request.form['frequentFlyer']
    if FrequentFlyer == 'Yes':
        FrequentFlyer = 1
    else:
        FrequentFlyer = 0
    
    EverTravelledAbroad = request.form['EverTravelledAbroad']
    if EverTravelledAbroad == 'Yes':
        EverTravelledAbroad = 1
    else:
        EverTravelledAbroad = 0

    # Create an array with the input features
    input_features = np.array([Age, EmployementType, GraduateOrNot, AnnualIncome, FamilyMembers, chronicDiseases, FrequentFlyer, EverTravelledAbroad])

    # Reshape the array for compatibility with MinMaxScaler
    input_features_reshaped = input_features.reshape(1, -1)

    # Scale the input features using the MinMaxScaler
    input_features_scaled = ms.transform(input_features_reshaped)

    # Make the prediction
    prediction = model.predict(input_features_scaled)

    # Determine the prediction text based on the model output
    if prediction[0] == 1:
        prediction_text = "The Person will take the Insurance" 
    else:
       prediction_text = "The Person does not take the Insurance"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the selected model
    model_choice = request.form['model']
    if model_choice == 'Random Forest':
        model = joblib.load('heartrf.pkl')
    elif model_choice == 'Logistic Regression':
        model = joblib.load('heartlg.pkl')
    else:
        model = joblib.load('heartdc.pkl')

    # Get the input features from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    bp = int(request.form['bp'])
    cholesterol = int(request.form['cholesterol'])
    fbs = int(request.form['fbs'])
    ekg = int(request.form['ekg'])
    max_hr = int(request.form['max_hr'])
    ex_angina = int(request.form['ex_angina'])
    st_depression = float(request.form['st_depression'])
    slope = int(request.form['slope'])
    vessels = int(request.form['vessels'])
    thallium = int(request.form['thallium'])

    # Make a prediction using the loaded model
    prediction = model.predict([[age, sex, cp, bp, cholesterol, fbs, ekg, max_hr, ex_angina, st_depression, slope, vessels, thallium]])
    if prediction[0] == 0:
        result = 'No heart disease'
    else:
        result = 'Heart disease detected'

    # Render the results page with the prediction
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

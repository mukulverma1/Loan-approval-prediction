from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Encode inputs
        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Yes" else 0
        education = 0 if education == "Graduate" else 1
        self_employed = 1 if self_employed == "Yes" else 0
        dependents = 3 if dependents == "3+" else int(dependents)
        area_dict = {"Rural": 0, "Semiurban": 1, "Urban": 2}
        property_area = area_dict.get(property_area, 0)

        features = [gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount,
                    loan_term, credit_history, property_area]

        final_input = np.array([features])
        prediction = model.predict(final_input)

        result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

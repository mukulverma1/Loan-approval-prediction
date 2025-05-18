from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate-graph')
def generate_graph():
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    os.makedirs('static/images', exist_ok=True)
    df = pd.read_csv('LoanApprovalPrediction.csv')

    df['Loan_Status'] = df['Loan_Status'].replace({'Y': 'Approved', 'N': 'Rejected'})
    df['Self_Employed'] = df['Self_Employed'].fillna('Unknown')
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Property_Area'] = df['Property_Area'].fillna('Unknown')

    ### 1. Loan Status Chart
    status_counts = df['Loan_Status'].value_counts()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(status_counts.index, status_counts.values, color=['#1f77b4', '#ff7f0e'])
    for bar, value in zip(bars, status_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{value}', ha='center')
    plt.title('Loan Status Distribution')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/images/loan_status_chart.png')
    plt.close()

    ### 2. Loan Status by Gender
    plt.figure(figsize=(6, 4))
    pd.crosstab(df['Gender'], df['Loan_Status']).plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'])
    plt.title('Loan Status by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/images/gender_chart.png')
    plt.close()

    ### 3. Loan Status by Self-Employment
    plt.figure(figsize=(6, 4))
    pd.crosstab(df['Self_Employed'], df['Loan_Status']).plot(kind='bar', stacked=True, color=['#9467bd', '#8c564b'])
    plt.title('Loan Status by Employment Type')
    plt.xlabel('Self Employed')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/images/employment_chart.png')
    plt.close()

    ### 4. Property Area vs Loan Status
    plt.figure(figsize=(6, 4))
    pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', stacked=True, color=['#17becf', '#bcbd22'])
    plt.title('Property Area vs Loan Status')
    plt.xlabel('Property Area')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/images/property_area_chart.png')
    plt.close()

    ### 5. Income vs Loan Amount
    plt.figure(figsize=(6, 4))
    plt.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.5, c='purple')
    plt.title('Income vs Loan Amount')
    plt.xlabel('Applicant Income')
    plt.ylabel('Loan Amount')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/images/income_scatter.png')
    plt.close()

    return render_template('generate-graph.html')

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

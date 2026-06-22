# Employee Burnout Prediction using Machine Learning


## Project Overview

This project predicts employee burnout rate using Machine Learning algorithms.

The system analyzes employee factors like stress level, working hours, workload and other features to predict burnout risk.


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit


## Machine Learning Models

Three models are implemented:

1. Linear Regression

2. Decision Tree Regressor

3. Random Forest Regressor


## Project Structure

Employee_Burnout_Project/

dataset/
- employee_burnout.csv

src/
- preprocessing.py
- train.py
- predict.py

model/
- burnout_model.pkl
- scaler.pkl

app.py


## Workflow

Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Model Training
↓
Model Comparison
↓
Burnout Prediction


## Features Analysed

- Age
- Gender
- Job Role
- Work Pressure
- Working Hours
- Sleep Hours


## How to Run


Install requirements:

pip install -r requirements.txt


Train model:

python src/train.py


Run application:

streamlit run app.py



## Output

The system predicts:

- Low Burnout Risk
- Medium Burnout Risk
- High Burnout Risk


## Conclusion

Random Forest model provides better performance compared to other models.

The system helps identify employees who may be at risk of burnout.
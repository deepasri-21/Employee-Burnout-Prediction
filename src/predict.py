import pickle
import pandas as pd


model = pickle.load(
    open(
        "model/burnout_model.pkl",
        "rb"
    )
)


scaler = pickle.load(
    open(
        "model/scaler.pkl",
        "rb"
    )
)


employee = pd.DataFrame([{

"Gender":1,
"Company Type":0,
"WFH Setup Available":1,
"Designation":2,
"Resource Allocation":3,
"Mental Fatigue Score":6

}])


employee_scaled = scaler.transform(
    employee
)


# Fix warning
employee_scaled = pd.DataFrame(
    employee_scaled,
    columns=employee.columns
)


prediction = model.predict(
    employee_scaled
)


rate = prediction[0]


print(
    "Predicted Burn Rate:",
    rate
)


if rate >=0.6:
    print("High Burnout Risk")

elif rate >=0.3:
    print("Medium Burnout Risk")

else:
    print("Low Burnout Risk")
import pickle
import numpy as np



# Load model

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




# New employee data


employee = np.array([

[
25,
1,
2,
3,
8,
6
]

])



# Scaling

employee = scaler.transform(
    employee
)



# Prediction

result = model.predict(
    employee
)



rate = result[0]



print(
    "Predicted Burn Rate:",
    rate
)



if rate >= 0.6:

    print(
        "High Burnout Risk"
    )


elif rate >= 0.3:

    print(
        "Medium Burnout Risk"
    )


else:

    print(
        "Low Burnout Risk"
    )
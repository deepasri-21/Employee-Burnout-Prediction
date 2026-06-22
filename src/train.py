from preprocessing import preprocess_data


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


import pickle
import numpy as np



# Load data

X, y, scaler = preprocess_data(
    "dataset/employee_burnout.csv"
)



# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Models

models = {

"Linear Regression":
LinearRegression(),


"Decision Tree":
DecisionTreeRegressor(
    random_state=42
),


"Random Forest":
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

}



best_model = None
best_score = 0



for name,model in models.items():


    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_test
    )


    score = r2_score(
        y_test,
        pred
    )


    print("----------------")
    print(name)

    print(
        "R2 Score:",
        score
    )


    print(
        "MAE:",
        mean_absolute_error(
            y_test,
            pred
        )
    )


    print(
        "RMSE:",
        np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )
    )



    if score > best_score:

        best_score = score
        best_model = model




# Save best model


pickle.dump(
    best_model,
    open(
        "model/burnout_model.pkl",
        "wb"
    )
)



pickle.dump(
    scaler,
    open(
        "model/scaler.pkl",
        "wb"
    )
)



print(
    "Best Model Saved"
)

print(
    "Best Accuracy:",
    best_score
)

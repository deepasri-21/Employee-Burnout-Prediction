import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)


    # Remove duplicates
    df = df.drop_duplicates()


    # Remove missing values
    df = df.dropna()


    # Remove unwanted columns
    remove_columns = [
        "Employee ID",
        "Date of Joining"
    ]


    for col in remove_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)



    # Encode categorical columns
    encoder = LabelEncoder()


    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])



    # Split feature and target

    X = df.drop(
        "Burn Rate",
        axis=1
    )


    y = df["Burn Rate"]



    # Scaling

    scaler = StandardScaler()


    X_scaled = scaler.fit_transform(
        X
    )


    return X_scaled, y, scaler
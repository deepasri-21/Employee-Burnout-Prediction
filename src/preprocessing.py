import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(file_path):

    df = pd.read_csv(file_path)


    # remove duplicates
    df = df.drop_duplicates()


    # remove missing values
    df = df.dropna()



    # remove unwanted columns

    remove_cols = [
        "Employee ID",
        "Date of Joining"
    ]


    for col in remove_cols:
        if col in df.columns:
            df.drop(
                col,
                axis=1,
                inplace=True
            )



    # Encode categorical columns

    encoder = LabelEncoder()


    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])



    # Features and target

    X = df.drop(
        "Burn Rate",
        axis=1
    )


    y = df["Burn Rate"]



    # Scaling with column names

    scaler = StandardScaler()


    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )


    return X_scaled, y, scaler
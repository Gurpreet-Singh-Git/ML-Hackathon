import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
train = pd.read_csv("data/train_dataset.csv")
test = pd.read_csv("data/test_dataset.csv")

def feature_engineering(df):

    df = df.copy()

    df["delta_T"] = (
        df["inlet_temperature_K"]
        - df["jacket_temperature_K"]
    )

    df["T_product"] = (
        df["inlet_temperature_K"]
        * df["jacket_temperature_K"]
    )

    df["flow_length"] = (
        df["flow_rate_L_min"]
        * df["length_m"]
    )

    df["length_per_flow"] = (
        df["length_m"]
        / df["flow_rate_L_min"].replace(0, np.nan)
    )

    df["concentration_Tin"] = (
        df["concentration_mol_L"]
        * df["inlet_temperature_K"]
    )

    df["concentration_Tj"] = (
        df["concentration_mol_L"]
        * df["jacket_temperature_K"]
    )

    df["concentration_flow"] = (
        df["concentration_mol_L"]
        * df["flow_rate_L_min"]
    )

    df["temp_diff"] = (
        df["jacket_temperature_K"]
        - df["inlet_temperature_K"]
    )

    return df


train = feature_engineering(train)
test = feature_engineering(test)

X = train.drop(columns="overall_yield")
y = train["overall_yield"]


test = test[X.columns]



X = X.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

medians = X.median()

X = X.fillna(medians)
test = test.fillna(medians)



model = ExtraTreesRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    max_depth=50,
    bootstrap=False,
    random_state=99,
    n_jobs=-1
)


model.fit(X, y)


predictions = model.predict(test)


submission = pd.DataFrame({
    "overall_yield": predictions
})

submission.to_csv(
    "Operators.csv",
    index=False
)



print("\nPredictions:")
print(submission)

print("\nNumber of predictions:", len(predictions))
print("Prediction min:", predictions.min())
print("Prediction max:", predictions.max())
print("Prediction mean:", predictions.mean())

print("\nOperators.csv created successfully!")
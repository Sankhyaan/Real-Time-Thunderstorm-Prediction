from pathlib import Path

import joblib
import pandas as pd

from metpy_indices import process_sounding

# ==========================================================
# Paths
# ==========================================================

MODEL_DIR = Path("final_outputs")

model = joblib.load(MODEL_DIR / "CatBoost_Final.pkl")
feature_order = joblib.load(MODEL_DIR / "feature_order.pkl")

probability_mapping = pd.read_csv(
    MODEL_DIR / "CatBoost_Operational_Probability_Mapping.csv"
)

# Highest probability first
probability_mapping = probability_mapping.sort_values(
    "Display_Percentage",
    ascending=False
).reset_index(drop=True)


# ==========================================================
# Predict
# ==========================================================

def predict(sounding):

    # ------------------------------------------------------
    # Calculate meteorological indices
    # ------------------------------------------------------

    feature_df, indices = process_sounding(sounding)

    feature_df = feature_df.reindex(
        columns=feature_order,
        fill_value=float("nan")
    )

    # ------------------------------------------------------
    # Raw CatBoost probability
    # ------------------------------------------------------

    raw_probability = model.predict_proba(feature_df)[0, 1]

    # ------------------------------------------------------
    # Operational probability mapping
    # ------------------------------------------------------

    operational_probability = 35

    for _, row in probability_mapping.iterrows():

        if raw_probability >= row["Min_Model_Probability"]:

            operational_probability = int(
                row["Display_Percentage"]
            )
            break

    # ------------------------------------------------------
    # Risk Category
    # ------------------------------------------------------

    if operational_probability == 95:
        risk = "VERY HIGH"

    elif operational_probability == 85:
        risk = "HIGH"

    elif operational_probability == 75:
        risk = "MODERATELY HIGH"

    elif operational_probability == 65:
        risk = "MODERATE"

    elif operational_probability == 55:
        risk = "MODERATELY LOW"

    elif operational_probability == 45:
        risk = "LOW"

    else:
        risk = "VERY LOW"

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return {
        "raw_probability": raw_probability * 100,
        "probability": operational_probability,
        "risk": risk,
        "indices": indices,
        "features": feature_df,
    }
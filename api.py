from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
from fastapi.responses import JSONResponse

import base64
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import uvicorn
import os
import matplotlib
matplotlib.use('Agg')

app = FastAPI()

# Load saved model, preprocessor and data
model = XGBClassifier()
X_train = joblib.load(r"resource/X_train_non_professional.joblib")
model.load_model(r"resource/xgb_non_professional_model.json")

model_doctor = XGBClassifier()
model_doctor.load_model(r"resource/xgb_model.json")
preprocessor = joblib.load(r"resource/preprocessor.pkl")
X_train_doctor = joblib.load(r"resource/X_train.joblib")

# Define expected order of columns
expected_cols_doctor = ['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'RestingBP',
                 'ChestPainType', 'ExerciseAngina', 'RestingECG', 'ST_Slope', 'Sex']

expected_cols = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "High_BP",
    "High_Cholesterol",
    "Diabetes",
    "Smoking",
    "Obesity",
    "Sedentary_Lifestyle",
    "Family_History",
    "Chronic_Stress",
    "Gender",
    "Age"
]

class InputDataDoctor(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

class InputData(BaseModel):
    Chest_Pain: float
    Shortness_of_Breath: float
    Fatigue: float
    Palpitations: float
    Dizziness: float
    Swelling: float
    Pain_Arms_Jaw_Back: float
    Cold_Sweats_Nausea: float
    High_BP: float
    High_Cholesterol: float
    Diabetes: float
    Smoking: float
    Obesity: float
    Sedentary_Lifestyle: float
    Family_History: float
    Chronic_Stress: float
    Gender: float
    Age: float

@app.post("/predict/doctor")
def predict_heart_disease_doctor(input_data: InputDataDoctor):
    # Convert input to DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    df = df[expected_cols_doctor]

    # Transform input
    transformed_data = preprocessor.transform(df)

    # Get feature names
    onehot_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_feature_names = onehot_encoder.get_feature_names_out(input_features=['ChestPainType', 'ExerciseAngina', 'RestingECG', 'ST_Slope', 'Sex'])
    num_feature_names = ['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'RestingBP']
    all_feature_names = num_feature_names + list(cat_feature_names)

    # SHAP explainer and values
    explainer = shap.Explainer(model_doctor, X_train_doctor)
    shap_values = explainer(transformed_data)
    shap_values.feature_names = all_feature_names

    # Create and save SHAP waterfall plot
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall Plot")
    plt.tight_layout()
    plot_path = "shap_waterfall_plot_doctor.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close()

    # Make prediction
    prediction = model_doctor.predict(transformed_data)[0]
    probability = model_doctor.predict_proba(transformed_data)[0][1]

    with open(plot_path, "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode("utf-8")
    
    return JSONResponse(content={
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "shap_plot": encoded_img
    })

@app.post("/predict")
def predict_heart_disease(input_data: InputData):
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    df = df[expected_cols]

    explainer = shap.Explainer(model,X_train)

    # Calculate SHAP values for the transformed data
    shap_values = explainer(df)


    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall Plot")
    plt.tight_layout()
    plot_path = "shap_waterfall_plot.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close()

    prediction = model.predict(df)[0]
    probability=model.predict_proba(df)[0][1]

    with open(plot_path, "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode("utf-8")
    
    return JSONResponse(content={
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "shap_plot": encoded_img
    })
    
# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)  
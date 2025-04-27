import joblib
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

loaded_model = XGBClassifier()
loaded_model.load_model(r"resource/xgb_model.json")

preprocessor = joblib.load(r"resource/preprocessor.pkl")
X_train_loaded = joblib.load(r"resource/X_train.joblib")

data = pd.DataFrame({
    'Age': [41],
    'Sex': ['M'],
    'ChestPainType': ['ATA'],
    'RestingBP': [140],
    'Cholesterol': [289],
    'FastingBS': [0],
    'RestingECG': ['Normal'],
    'MaxHR': [140],
    'ExerciseAngina': ['N'],
    'Oldpeak': [0],
    'ST_Slope': ['Down']
})

expected_cols = ['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'RestingBP','ChestPainType', 'ExerciseAngina', 'RestingECG', 'ST_Slope', 'Sex']
data = data[expected_cols]

transformed_data = preprocessor.transform(data) #transform the data 

#transform the categorical column so that it can replace the features in SHAP with the real name
onehot_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
categorical_feature_names = onehot_encoder.get_feature_names_out(input_features=['ChestPainType', 'ExerciseAngina', 'RestingECG', 'ST_Slope', 'Sex'])
numerical_feature_names = ['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'RestingBP']
all_feature_names = numerical_feature_names + list(categorical_feature_names)

explainer = shap.Explainer(loaded_model,X_train_loaded)
shap_values = explainer(transformed_data)
shap_values.feature_names = all_feature_names

plt.clf()
shap.plots.waterfall(shap_values[0],show=False) 
plt.title("SHAP Waterfall Plot for Prediction")
plt.tight_layout()
plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=300, pad_inches=0.2)
plt.close()

predictions = loaded_model.predict(transformed_data)
print(predictions)
probability=loaded_model.predict_proba(transformed_data)
print(probability)
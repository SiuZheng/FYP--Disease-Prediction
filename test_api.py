import requests
import base64

url = "http://127.0.0.1:8000/predict"

# Same format as the DataFrame input
data = {
    "Chest_Pain": 1,
    "Shortness_of_Breath": 1,
    "Fatigue": 0,
    "Palpitations": 1,
    "Dizziness": 0,
    "Swelling": 1,
    "Pain_Arms_Jaw_Back": 1,
    "Cold_Sweats_Nausea": 0,
    "High_BP": 1,
    "High_Cholesterol": 1,
    "Diabetes": 0,
    "Smoking": 1,
    "Obesity": 1,
    "Sedentary_Lifestyle": 1,
    "Family_History": 1,
    "Chronic_Stress": 1,
    "Gender": 1,
    "Age": 55
}

response = requests.post(url, json=data)
content = response.json()
'''
{
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "shap_plot": encoded_img(base64)
}
'''
image_data = base64.b64decode(content["shap_plot"])
with open("decoded_image.png", "wb") as img_file:
    img_file.write(image_data)

# Print the full response
print("Status Code:", response.status_code)
print("Response JSON:", response.json()["prediction"])
print("probability:", response.json()["probability"])
package com.example.as_heartdiseaseprediction.data.model

data class PredictionRequest(
    val Chest_Pain: Int,
    val Shortness_of_Breath: Int,
    val Fatigue: Int,
    val Palpitations: Int,
    val Dizziness: Int,
    val Swelling: Int,
    val Pain_Arms_Jaw_Back: Int,
    val Cold_Sweats_Nausea: Int,
    val High_BP: Int,
    val High_Cholesterol: Int,
    val Diabetes: Int,
    val Smoking: Int,
    val Obesity: Int,
    val Sedentary_Lifestyle: Int,
    val Family_History: Int,
    val Chronic_Stress: Int,
    val Gender: Int,
    val Age: Int
)

data class PredictionResponse(
    val prediction: Int,
    val probability: Float,
    val shap_plot: String,
    val explanation: String
) 
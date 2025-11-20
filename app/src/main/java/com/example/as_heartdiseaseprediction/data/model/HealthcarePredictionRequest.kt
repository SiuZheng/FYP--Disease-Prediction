package com.example.as_heartdiseaseprediction.data.model

data class HealthcarePredictionRequest(
    val Age: Int,
    val Sex: String, // 'M' or 'F'
    val ChestPainType: String, // 'TA', 'ATA', 'NAP', 'ASY'
    val RestingBP: Int,
    val FastingBS: Int, // 1 if > 120, else 0
    val Cholesterol: Int,
    val RestingECG: String, // 'Normal', 'ST', 'LVH'
    val MaxHR: Int,
    val ExerciseAngina: String, // 'Y' or 'N'
    val Oldpeak: Int,
    val ST_Slope: String // 'UP', 'Flat', 'Down'
) 
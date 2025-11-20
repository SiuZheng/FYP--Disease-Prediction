package com.example.as_heartdiseaseprediction.data.model

data class PredictionEntry(
    val userId: String = "",
    val userName: String = "",
    val timestamp: Long = 0L,
    val probability: Float = 0f,
    val data: Any? = null // store raw map for flexibility
)


package com.example.as_heartdiseaseprediction.config

object Config {
    // API Configuration
    const val API_BASE_URL = "https://a09772a86320.ngrok-free.app" // Replace this with your actual API URL
    const val PREDICT_ENDPOINT = "predict"
    const val DOCTOR_PREDICT_ENDPOINT = "predict/doctor"
    const val CHAT_ENDPOINT = "chat"
    // API Timeouts (in seconds)
    const val CONNECT_TIMEOUT = 60L
    const val READ_TIMEOUT = 60L
    const val WRITE_TIMEOUT = 60L
} 
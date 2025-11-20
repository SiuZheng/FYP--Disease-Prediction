package com.example.as_heartdiseaseprediction.data.api

import com.example.as_heartdiseaseprediction.config.Config
import com.example.as_heartdiseaseprediction.data.model.HealthcarePredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionResponse
import com.example.as_heartdiseaseprediction.data.model.ChatRequest
import com.example.as_heartdiseaseprediction.data.model.ChatResponse
import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {
    @POST(Config.PREDICT_ENDPOINT)
    suspend fun predictHeartDisease(@Body request: PredictionRequest): PredictionResponse

    @POST(Config.DOCTOR_PREDICT_ENDPOINT)
    suspend fun predictHeartDiseaseDoctor(@Body request: HealthcarePredictionRequest): PredictionResponse

    @POST(Config.CHAT_ENDPOINT)
    suspend fun chat(@Body request: ChatRequest): ChatResponse

}

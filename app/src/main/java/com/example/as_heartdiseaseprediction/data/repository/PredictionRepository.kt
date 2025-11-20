package com.example.as_heartdiseaseprediction.data.repository

import com.example.as_heartdiseaseprediction.data.api.ApiClient
import com.example.as_heartdiseaseprediction.data.model.HealthcarePredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionResponse

class PredictionRepository {
    private val apiService = ApiClient.apiService

    suspend fun predictHeartDisease(request: PredictionRequest): Result<PredictionResponse> {
        return try {
            val response = apiService.predictHeartDisease(request)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun predictHeartDiseaseDoctor(request: HealthcarePredictionRequest): Result<PredictionResponse> {
        return try {
            val response = apiService.predictHeartDiseaseDoctor(request)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
} 
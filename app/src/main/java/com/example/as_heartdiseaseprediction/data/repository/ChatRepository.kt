package com.example.as_heartdiseaseprediction.data.repository


import com.example.as_heartdiseaseprediction.data.api.ApiClient
import com.example.as_heartdiseaseprediction.data.model.ChatRequest
import com.example.as_heartdiseaseprediction.data.model.ChatResponse


class ChatRepository {
    private val apiService = ApiClient.apiService

    suspend fun chat(request: ChatRequest): Result<ChatResponse>{
        return try {
            val response = apiService.chat(request)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

}
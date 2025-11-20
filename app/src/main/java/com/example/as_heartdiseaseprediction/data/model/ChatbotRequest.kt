package com.example.as_heartdiseaseprediction.data.model

data class ChatRequest(
    val user_message: String,
    val conversation_id: String? = null
)

data class ChatResponse(
    val conversation_id: String,
    val answer: String
)
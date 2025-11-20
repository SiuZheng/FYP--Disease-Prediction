package com.example.as_heartdiseaseprediction.ui.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.as_heartdiseaseprediction.data.model.ChatRequest
import com.example.as_heartdiseaseprediction.data.repository.ChatRepository
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import com.google.firebase.firestore.Query

class ChatViewModel : ViewModel() {
    private val repository = ChatRepository()
    private val _chatResponse = MutableLiveData<String>()
    val chatResponse: LiveData<String> get() = _chatResponse

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> get() = _isLoading

    private var conversationId: String? = null

    private val db = FirebaseFirestore.getInstance()
    private val auth = FirebaseAuth.getInstance()

    fun sendMessage(message: String) {
        viewModelScope.launch {
            _isLoading.value = true
            val userId = auth.currentUser?.uid

            if (userId != null && conversationId == null) {
                // Step 1: Retrieve existing conversationId from Firestore
                try {
                    val snapshot = db.collection("conversations")
                        .whereEqualTo("userId", userId)
                        .limit(1)
                        .get()
                        .await() // Use kotlinx-coroutines-play-services for await()

                    if (!snapshot.isEmpty) {
                        // If a conversation is found, set conversationId
                        conversationId = snapshot.documents.first().getString("conversationId")
                        println("Fetched existing conversationId from Firestore: $conversationId")
                    }
                } catch (e: Exception) {
                    println("Error fetching conversationId: ${e.localizedMessage}")
                }
            }

            // Step 2: Proceed with sending message
            val request = ChatRequest(
                user_message = message,
                conversation_id = conversationId // This will be null if no existing conversationId found
            )

            val result = repository.chat(request)

            result.onSuccess { response ->
                conversationId = response.conversation_id // Update with new conversationId
                _chatResponse.value = response.answer

                if (userId != null && response.conversation_id != null) {
                    // Step 3: Save the new or updated conversationId to Firestore
                    val data = hashMapOf(
                        "userId" to userId,
                        "conversationId" to response.conversation_id,
                        "timestamp" to System.currentTimeMillis()
                    )

                    db.collection("conversations")
                        .add(data)
                        .addOnSuccessListener { document ->
                            println("Conversation saved with ID: ${document.id}")
                        }
                        .addOnFailureListener { e ->
                            println("Error saving conversation: ${e.localizedMessage}")
                        }
                } else {
                    println("User not authenticated or conversationId is null.")
                }
            }.onFailure { error ->
                _chatResponse.value = "Error: ${error.localizedMessage}"
            }

            _isLoading.value = false
        }
    }
}
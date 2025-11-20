package com.example.as_heartdiseaseprediction.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.as_heartdiseaseprediction.data.model.HealthcarePredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionResponse
import com.example.as_heartdiseaseprediction.data.model.PredictionEntry
import com.example.as_heartdiseaseprediction.data.repository.PredictionRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore


class PredictionViewModel : ViewModel() {
    private val repository = PredictionRepository()
    
    private val _predictionResult = MutableStateFlow<Result<PredictionResponse>?>(null)
    val predictionResult: StateFlow<Result<PredictionResponse>?> = _predictionResult.asStateFlow()
    
    private val _shapPlot = MutableStateFlow<String?>(null)
    val shapPlot: StateFlow<String?> = _shapPlot
    
    private val _explanation = MutableStateFlow<String?>(null)
    val explanation: StateFlow<String?> = _explanation
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    private val _username = MutableStateFlow("User")
    val username: StateFlow<String> = _username.asStateFlow()

    private val db = FirebaseFirestore.getInstance()
    private val auth = FirebaseAuth.getInstance()

    private val _history = MutableStateFlow<List<PredictionEntry>>(emptyList())
    val history: StateFlow<List<PredictionEntry>> = _history

    init {
        println("Debug - PredictionViewModel: Initialized with username: ${_username.value}")
    }

    fun fetchPredictionHistory() {
        _isLoading.value = true
        val userId = auth.currentUser?.uid ?: return

        val doctorRef = db.collection("model-doctor").document(userId).collection("predictions")
        val generalRef = db.collection("model-general-user").document(userId).collection("predictions")

        val combinedPredictions = mutableListOf<PredictionEntry>()

        doctorRef.get()
            .addOnSuccessListener { doctorSnapshot ->
                val doctorPredictions = doctorSnapshot.documents.mapNotNull { doc ->
                    val dataMap = doc.data
                    println("Data Map%%%%%%%%%%%%%%%: $dataMap")
                    val rawData = dataMap?.get("data") as? Map<String, Any>
                    val request = rawData?.let { mapToHealthcarePredictionRequest(it) }
                    println("NOOOOOOOOOOOO$request")
                    request?.let {
                        PredictionEntry(
                            userId = dataMap["userId"] as? String ?: "",
                            userName = dataMap["userName"] as? String ?: "",
                            timestamp = (dataMap["timestamp"] as? Number)?.toLong() ?: 0L,
                            probability = (dataMap["probability"] as? Number)?.toFloat() ?: 0f,
                            data = it
                        )
                    }
                }
                combinedPredictions.addAll(doctorPredictions)

                generalRef.get()
                    .addOnSuccessListener { generalSnapshot ->
                        val generalPredictions = generalSnapshot.documents.mapNotNull { doc ->
                            val dataMap = doc.data
                            println("Data Map%%%%%%%%%%%%%%%: $dataMap")
                            val rawData = dataMap?.get("data") as? Map<String, Any>
                            val request = rawData?.let { mapToPredictionRequest(it) }

                            request?.let {
                                PredictionEntry(
                                    userId = dataMap["userId"] as? String ?: "",
                                    userName = dataMap["userName"] as? String ?: "",
                                    timestamp = (dataMap["timestamp"] as? Number)?.toLong() ?: 0L,
                                    probability = (dataMap["probability"] as? Number)?.toFloat() ?: 0f,
                                    data = it
                                )
                            }
                        }
                        combinedPredictions.addAll(generalPredictions)
                        _history.value = combinedPredictions.sortedByDescending { it.timestamp }
                        _isLoading.value = false
                    }
                    .addOnFailureListener { e ->
                        println("Error fetching general history: $e")
                        _isLoading.value = false
                    }
            }
            .addOnFailureListener { e ->
                println("Error fetching doctor history: $e")
                _isLoading.value = false
            }
    }

    fun savePredictionData(request: PredictionRequest, userName: String,result: Float ) {
        val userId = auth.currentUser?.uid ?: "unknown_user"

        val data = hashMapOf(
            "userId" to userId,
            "userName" to userName,
            "timestamp" to System.currentTimeMillis(),
            "data" to request,// will store fields like Gender, Age, etc.
            "probability" to result
        )

        db.collection("model-general-user")
            .document(userId)
            .collection("predictions")
            .add(data)
            .addOnSuccessListener { documentRef ->
                println("Data saved with ID: ${documentRef.id}")
            }
            .addOnFailureListener { e ->
                println("Error saving data: $e")
            }
    }

    fun savePredictionDataDoctor(request: HealthcarePredictionRequest, userName: String,result: Float ) {
        val userId = auth.currentUser?.uid ?: "unknown_user"
        println("Debug - PredictionViewModel: UserID: $userId")
        val data = hashMapOf(
            "userId" to userId,
            "userName" to userName,
            "timestamp" to System.currentTimeMillis(),
            "data" to request,// will store fields like Gender, Age, etc.
            "probability" to result
        )

        db.collection("model-doctor")
            .document(userId)
            .collection("predictions")
            .add(data)
            .addOnSuccessListener { documentRef ->
                println("Data saved with ID: ${documentRef.id}")
            }
            .addOnFailureListener { e ->
                println("Error saving data: $e")
            }
    }

    fun predictHeartDisease(request: PredictionRequest) {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val result = repository.predictHeartDisease(request)
                _predictionResult.value = result
                result.onSuccess { response ->
                    _shapPlot.value = response.shap_plot
                    _explanation.value = response.explanation
                    savePredictionData(request, _username.value,response.probability)
                }
            } catch (e: Exception) {
                _error.value = e.message ?: "An unknown error occurred"
                _predictionResult.value = Result.failure(e)
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    fun predictHeartDiseaseDoctor(request: HealthcarePredictionRequest) {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val result = repository.predictHeartDiseaseDoctor(request)
                _predictionResult.value = result
                result.onSuccess { response ->
                    _shapPlot.value = response.shap_plot
                    _explanation.value = response.explanation
                    savePredictionDataDoctor(request, _username.value,response.probability)
                }
            } catch (e: Exception) {
                _error.value = e.message ?: "An unknown error occurred"
                _predictionResult.value = Result.failure(e)
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    fun setUsername(name: String) {
        println("Debug - PredictionViewModel: Setting username from ${_username.value} to $name")
        _username.value = name
        println("Debug - PredictionViewModel: Username after setting: ${_username.value}")
    }
    
    fun clearState() {
        println("Debug - PredictionViewModel: Clearing state, username was: ${_username.value}")
        _predictionResult.value = null
        _shapPlot.value = null
        _explanation.value = null
        _error.value = null
        _isLoading.value = false
        _username.value = "User"
        println("Debug - PredictionViewModel: State cleared, username is now: ${_username.value}")
    }
    private fun mapToPredictionRequest(data: Map<String, Any>?): PredictionRequest {
        return PredictionRequest(
            Chest_Pain = (data?.get("chest_Pain") as? Number)?.toInt() ?: 0,
            Shortness_of_Breath = (data?.get("shortness_of_Breath") as? Number)?.toInt() ?: 0,
            Fatigue = (data?.get("fatigue") as? Number)?.toInt() ?: 0,
            Palpitations = (data?.get("palpitations") as? Number)?.toInt() ?: 0,
            Dizziness = (data?.get("dizziness") as? Number)?.toInt() ?: 0,
            Swelling = (data?.get("swelling") as? Number)?.toInt() ?: 0,
            Pain_Arms_Jaw_Back = (data?.get("pain_Arms_Jaw_Back") as? Number)?.toInt() ?: 0,
            Cold_Sweats_Nausea = (data?.get("cold_Sweats_Nausea") as? Number)?.toInt() ?: 0,
            High_BP = (data?.get("high_BP") as? Number)?.toInt() ?: 0,
            High_Cholesterol = (data?.get("high_Cholesterol") as? Number)?.toInt() ?: 0,
            Diabetes = (data?.get("diabetes") as? Number)?.toInt() ?: 0,
            Smoking = (data?.get("smoking") as? Number)?.toInt() ?: 0,
            Obesity = (data?.get("obesity") as? Number)?.toInt() ?: 0,
            Sedentary_Lifestyle = (data?.get("sedentary_Lifestyle") as? Number)?.toInt() ?: 0,
            Family_History = (data?.get("family_History") as? Number)?.toInt() ?: 0,
            Chronic_Stress = (data?.get("chronic_Stress") as? Number)?.toInt() ?: 0,
            Gender = (data?.get("gender") as? Number)?.toInt() ?: 0,
            Age = (data?.get("age") as? Number)?.toInt() ?: 0
        )
    }
    private fun mapToHealthcarePredictionRequest(data: Map<String, Any>?): HealthcarePredictionRequest {
        return HealthcarePredictionRequest(
            Age = (data?.get("age") as? Number)?.toInt() ?: 0,
            Sex = data?.get("sex") as? String ?: "",
            ChestPainType = data?.get("chestPainType") as? String ?: "",
            RestingBP = (data?.get("restingBP") as? Number)?.toInt() ?: 0,
            FastingBS = (data?.get("fastingBS") as? Number)?.toInt() ?: 0,
            Cholesterol = (data?.get("cholesterol") as? Number)?.toInt() ?: 0,
            RestingECG = data?.get("restingECG") as? String ?: "",
            MaxHR = (data?.get("maxHR") as? Number)?.toInt() ?: 0,
            ExerciseAngina = data?.get("exerciseAngina") as? String ?: "",
            Oldpeak = (data?.get("oldpeak") as? Number)?.toInt() ?: 0,
            ST_Slope = data?.get("st_Slope") as? String ?: ""
        )
    }
}
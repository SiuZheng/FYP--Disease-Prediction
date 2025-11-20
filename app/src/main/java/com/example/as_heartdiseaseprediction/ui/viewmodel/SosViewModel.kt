package com.example.as_heartdiseaseprediction.ui.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.as_heartdiseaseprediction.data.service.SosService
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class SosUiState(
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val emergencyMessage: String? = null,
    val showConfirmationDialog: Boolean = false,
    val showCountdownDialog: Boolean = false,
    val countdownValue: Int = 0,
    val isCountdownActive: Boolean = false,
    val successMessage: String? = null
)

class SosViewModel(private val context: Context) : ViewModel() {
    private val sosService = SosService(context)
    
    private val _uiState = MutableStateFlow(SosUiState())
    val uiState: StateFlow<SosUiState> = _uiState.asStateFlow()
    
    fun prepareEmergencyMessage() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)
            
            sosService.getEmergencyMessage()
                .onSuccess { message ->
                    _uiState.value = _uiState.value.copy(
                        emergencyMessage = message,
                        isLoading = false
                    )
                    
                    // Check if auto-send is enabled
                    if (sosService.isAutoSendEnabled()) {
                        startCountdown()
                    } else {
                        _uiState.value = _uiState.value.copy(showConfirmationDialog = true)
                    }
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        errorMessage = exception.message ?: "Failed to prepare emergency message"
                    )
                    
                    // Show confirmation dialog even on error
                    _uiState.value = _uiState.value.copy(showConfirmationDialog = true)
                }
        }
    }
    
    private fun startCountdown() {
        viewModelScope.launch {
            val timerDelay = sosService.getTimerDelay()
            _uiState.value = _uiState.value.copy(
                showCountdownDialog = true,
                countdownValue = timerDelay,
                isCountdownActive = true
            )
            
            // Countdown loop
            for (i in timerDelay downTo 1) {
                if (!_uiState.value.isCountdownActive) break
                _uiState.value = _uiState.value.copy(countdownValue = i)
                delay(1000)
            }
            
            // Auto-send if countdown completed
            if (_uiState.value.isCountdownActive) {
                sendEmergencySms()
            }
        }
    }
    
    fun cancelCountdown() {
        _uiState.value = _uiState.value.copy(
            showCountdownDialog = false,
            isCountdownActive = false,
            countdownValue = 0
        )
    }
    
    fun hideConfirmationDialog() {
        _uiState.value = _uiState.value.copy(showConfirmationDialog = false)
    }
    
    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }
    
    fun clearSuccessMessage() {
        _uiState.value = _uiState.value.copy(successMessage = null)
    }
    
    fun sendEmergencySms() {
        viewModelScope.launch {
            val message = _uiState.value.emergencyMessage ?: "(From Heart Disease Prediction App) Emergency! Please help me. Location unavailable."
            
            // Always use SmsManager for direct sending (both auto-send and manual confirmation)
            sosService.sendSmsDirectly(message)
                .onSuccess { successMsg ->
                    _uiState.value = _uiState.value.copy(
                        showCountdownDialog = false,
                        showConfirmationDialog = false,
                        isCountdownActive = false,
                        successMessage = successMsg
                    )
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        showCountdownDialog = false,
                        showConfirmationDialog = false,
                        isCountdownActive = false,
                        errorMessage = "Failed to send SMS: ${exception.message}"
                    )
                }
        }
    }
    
    fun getEmergencyPhoneNumber(): String = sosService.getEmergencyNumber()
    
    fun getSmsIntent() = sosService.createEmergencySmsIntent()
    
    fun getSmsIntentWithLocation(latitude: Double, longitude: Double) = 
        sosService.createEmergencySmsIntentWithLocation(latitude, longitude)
}

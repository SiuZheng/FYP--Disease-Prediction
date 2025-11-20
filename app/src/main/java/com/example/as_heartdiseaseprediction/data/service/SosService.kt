package com.example.as_heartdiseaseprediction.data.service

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.location.Location
import android.net.Uri
import android.telephony.SmsManager
import com.example.as_heartdiseaseprediction.data.preferences.SosPreferences
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class SosService(private val context: Context) {
    
    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)
    private val sosPreferences = SosPreferences(context)
    
    companion object {
        private const val EMERGENCY_MESSAGE_WITH_LOCATION = "(From Heart Disease Prediction App) Emergency! Please help me. My location: "
        private const val EMERGENCY_MESSAGE_NO_LOCATION = "(From Heart Disease Prediction App) Emergency! Please help me. Location unavailable."
    }
    
    @SuppressLint("MissingPermission")
    suspend fun getCurrentLocation(): Result<Location> = suspendCancellableCoroutine { continuation ->
        fusedLocationClient.lastLocation
            .addOnSuccessListener { location: Location? ->
                if (location != null) {
                    continuation.resume(Result.success(location))
                } else {
                    continuation.resume(Result.failure(Exception("Unable to get current location: Last location is null")))
                }
            }
            .addOnFailureListener { e ->
                continuation.resumeWithException(e)
            }
    }
    
    fun createEmergencySmsIntent(): Intent {
        val emergencyNumber = sosPreferences.emergencyNumber
        return Intent(Intent.ACTION_VIEW).apply {
            data = Uri.parse("smsto:$emergencyNumber")
            putExtra("sms_body", EMERGENCY_MESSAGE_NO_LOCATION)
        }
    }
    
    fun createEmergencySmsIntentWithLocation(latitude: Double, longitude: Double): Intent {
        val emergencyNumber = sosPreferences.emergencyNumber
        val locationUrl = "https://maps.google.com/?q=$latitude,$longitude"
        val message = "$EMERGENCY_MESSAGE_WITH_LOCATION$locationUrl"
        
        return Intent(Intent.ACTION_VIEW).apply {
            data = Uri.parse("smsto:$emergencyNumber")
            putExtra("sms_body", message)
        }
    }
    
    suspend fun getEmergencyMessage(): Result<String> {
        return try {
            val locationResult = getCurrentLocation()
            if (locationResult.isSuccess) {
                val location = locationResult.getOrThrow()
                val locationUrl = "https://maps.google.com/?q=${location.latitude},${location.longitude}"
                Result.success("$EMERGENCY_MESSAGE_WITH_LOCATION$locationUrl")
            } else {
                Result.success(EMERGENCY_MESSAGE_NO_LOCATION)
            }
        } catch (e: Exception) {
            Result.success(EMERGENCY_MESSAGE_NO_LOCATION)
        }
    }
    
    @SuppressLint("MissingPermission")
    fun sendSmsDirectly(message: String): Result<String> {
        return try {
            val emergencyNumber = sosPreferences.emergencyNumber
            val smsManager = SmsManager.getDefault()
            smsManager.sendTextMessage(emergencyNumber, null, message, null, null)
            Result.success("✅ SOS sent successfully to $emergencyNumber")
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    fun getEmergencyNumber(): String = sosPreferences.emergencyNumber
    
    fun isAutoSendEnabled(): Boolean = sosPreferences.isAutoSendEnabled
    
    fun getTimerDelay(): Int = sosPreferences.timerDelay
}

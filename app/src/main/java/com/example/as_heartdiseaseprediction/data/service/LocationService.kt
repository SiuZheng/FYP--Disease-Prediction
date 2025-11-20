package com.example.as_heartdiseaseprediction.data.service

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class LocationResult(
    val latitude: Double,
    val longitude: Double,
    val accuracy: Float?
)

class LocationService(private val context: Context) {
    private val fusedLocationClient: FusedLocationProviderClient = 
        LocationServices.getFusedLocationProviderClient(context)
    
    private fun hasLocationPermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED || 
        ActivityCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    suspend fun getCurrentLocation(): Result<LocationResult> {
        if (!hasLocationPermission()) {
            return Result.failure(SecurityException("Location permission not granted"))
        }
        
        return try {
            val location = getLastKnownLocation()
            Result.success(location)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    private suspend fun getLastKnownLocation(): LocationResult = suspendCancellableCoroutine { continuation ->
        val cancellationTokenSource = CancellationTokenSource()
        
        fusedLocationClient.getCurrentLocation(
            Priority.PRIORITY_HIGH_ACCURACY,
            cancellationTokenSource.token
        ).addOnSuccessListener { location: Location? ->
            if (location != null) {
                continuation.resume(
                    LocationResult(
                        latitude = location.latitude,
                        longitude = location.longitude,
                        accuracy = location.accuracy
                    )
                )
            } else {
                // Fallback to last known location
                fusedLocationClient.lastLocation.addOnSuccessListener { lastLocation: Location? ->
                    if (lastLocation != null) {
                        continuation.resume(
                            LocationResult(
                                latitude = lastLocation.latitude,
                                longitude = lastLocation.longitude,
                                accuracy = lastLocation.accuracy
                            )
                        )
                    } else {
                        continuation.resumeWithException(
                            Exception("Unable to get current location")
                        )
                    }
                }.addOnFailureListener { exception ->
                    continuation.resumeWithException(exception)
                }
            }
        }.addOnFailureListener { exception ->
            continuation.resumeWithException(exception)
        }
        
        continuation.invokeOnCancellation {
            cancellationTokenSource.cancel()
        }
    }
}

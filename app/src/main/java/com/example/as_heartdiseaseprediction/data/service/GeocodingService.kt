package com.example.as_heartdiseaseprediction.data.service

import android.content.Context
import com.example.as_heartdiseaseprediction.data.api.GeocodingApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class GeocodingResult(
    val placeName: String,
    val formattedAddress: String?
)

class GeocodingService(private val context: Context) {
    private val geocodingApiService = GeocodingApiClient.geocodingApiService
    
    suspend fun getPlaceName(latitude: Double, longitude: Double): Result<GeocodingResult> = withContext(Dispatchers.IO) {
        try {
            val response = geocodingApiService.reverseGeocode(
                latlng = "$latitude,$longitude",
                key = "AIzaSyDHn03_dCpJks2vQ-eRMR_TjQp67bM4UTY"
            )
            
            if (response.status == "OK" && response.results.isNotEmpty()) {
                val result = response.results.first()
                val placeName = result.addressComponents?.find { 
                    it.types.contains("locality") || it.types.contains("administrative_area_level_1")
                }?.longName ?: result.formattedAddress ?: "Unknown Location"
                
                Result.success(
                    GeocodingResult(
                        placeName = placeName,
                        formattedAddress = result.formattedAddress
                    )
                )
            } else {
                Result.failure(Exception("Geocoding failed: ${response.status}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

package com.example.as_heartdiseaseprediction.data.repository

import android.content.Context
import com.example.as_heartdiseaseprediction.data.api.DistanceMatrixApiClient
import com.example.as_heartdiseaseprediction.data.api.PlaceDetailsApiClient
import com.example.as_heartdiseaseprediction.data.api.PlacesApiClient
import com.example.as_heartdiseaseprediction.data.model.Hospital
import com.example.as_heartdiseaseprediction.data.service.LocationService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class PlacesRepository(private val context: Context) {
    private val placesApiService = PlacesApiClient.placesApiService
    private val distanceMatrixApiService = DistanceMatrixApiClient.distanceMatrixApiService
    private val placeDetailsApiService = PlaceDetailsApiClient.placeDetailsApiService
    private val locationService = LocationService(context)
    
    suspend fun getNearbyHospitals(): Result<List<Hospital>> = withContext(Dispatchers.IO) {
        try {
            // Get current location
            val locationResult = locationService.getCurrentLocation()
            if (locationResult.isFailure) {
                return@withContext Result.failure(
                    Exception("Location Error: ${locationResult.exceptionOrNull()?.message ?: "Unable to get current location"}")
                )
            }
            
            val location = locationResult.getOrThrow()
            val locationString = "${location.latitude},${location.longitude}"
            
            val response = placesApiService.getNearbyHospitals(
                location = locationString,
                radius = 5000, // 5km radius
                type = "hospital",
                key = "AIzaSyDHn03_dCpJks2vQ-eRMR_TjQp67bM4UTY"
            )
            
            if (response.status == "OK") {
                val hospitals = response.results
                // Fetch distance and duration for each hospital
                val hospitalsWithDistance = getHospitalsWithDistance(locationString, hospitals)
                // Fetch place details for each hospital
                val hospitalsWithDetails = getHospitalsWithDetails(hospitalsWithDistance)
                Result.success(hospitalsWithDetails)
            } else {
                Result.failure(Exception("API Error: ${response.errorMessage ?: response.status}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getCurrentLocationString(): Result<String> = withContext(Dispatchers.IO) {
        try {
            val locationResult = locationService.getCurrentLocation()
            if (locationResult.isFailure) {
                return@withContext Result.failure(
                    Exception("Location Error: ${locationResult.exceptionOrNull()?.message ?: "Unable to get current location"}")
                )
            }
            
            val location = locationResult.getOrThrow()
            Result.success("${location.latitude},${location.longitude}")
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    private suspend fun getHospitalsWithDistance(origin: String, hospitals: List<Hospital>): List<Hospital> {
        if (hospitals.isEmpty()) return hospitals
        
        try {
            // Create destinations string from hospital coordinates
            val destinations = hospitals.mapNotNull { hospital ->
                hospital.geometry?.location?.let { location ->
                    "${location.lat},${location.lng}"
                }
            }.joinToString("|")
            
            if (destinations.isEmpty()) return hospitals
            
            // Call Distance Matrix API
            val distanceResponse = distanceMatrixApiService.getDistanceMatrix(
                origins = origin,
                destinations = destinations,
                key = "AIzaSyDHn03_dCpJks2vQ-eRMR_TjQp67bM4UTY",
                units = "metric"
            )
            
            if (distanceResponse.status == "OK" && distanceResponse.rows.isNotEmpty()) {
                val elements = distanceResponse.rows.first().elements
                
                return hospitals.mapIndexed { index, hospital ->
                    if (index < elements.size && elements[index].status == "OK") {
                        val element = elements[index]
                        hospital.copy(
                            distance = element.distance?.text,
                            duration = element.duration?.text
                        )
                    } else {
                        hospital
                    }
                }
            }
        } catch (e: Exception) {
            // If distance matrix fails, return hospitals without distance data
            // This ensures the app still works even if distance API fails
        }
        
        return hospitals
    }
    
    private suspend fun getHospitalsWithDetails(hospitals: List<Hospital>): List<Hospital> {
        return hospitals.map { hospital ->
            try {
                val placeDetailsResponse = placeDetailsApiService.getPlaceDetails(
                    placeId = hospital.placeId,
                    fields = "name,formatted_address,formatted_phone_number,international_phone_number,website,opening_hours",
                    key = "AIzaSyDHn03_dCpJks2vQ-eRMR_TjQp67bM4UTY"
                )
                
                if (placeDetailsResponse.status == "OK" && placeDetailsResponse.result != null) {
                    val details = placeDetailsResponse.result
                    hospital.copy(
                        name = details.name ?: hospital.name,
                        vicinity = details.formattedAddress ?: hospital.vicinity,
                        formattedPhoneNumber = details.formattedPhoneNumber ?: hospital.formattedPhoneNumber,
                        website = details.website ?: hospital.website,
                        openingHours = details.openingHours?.let { placeDetailsHours ->
                            com.example.as_heartdiseaseprediction.data.model.OpeningHours(
                                openNow = placeDetailsHours.openNow,
                                weekdayText = placeDetailsHours.weekdayText
                            )
                        } ?: hospital.openingHours
                    )
                } else {
                    hospital
                }
            } catch (e: Exception) {
                // If place details fails, return hospital without additional details
                hospital
            }
        }
    }
}

package com.example.as_heartdiseaseprediction.ui.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.as_heartdiseaseprediction.data.model.Hospital
import com.example.as_heartdiseaseprediction.data.repository.PlacesRepository
import com.example.as_heartdiseaseprediction.data.service.GeocodingService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class PlacesUiState(
    val hospitals: List<Hospital> = emptyList(),
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isEmpty: Boolean = false,
    val currentLocation: String? = null,
    val currentPlaceName: String? = null,
    val isLocationPermissionGranted: Boolean = false
)

class PlacesViewModel(private val context: Context) : ViewModel() {
    private val placesRepository = PlacesRepository(context)
    private val geocodingService = GeocodingService(context)
    
    private val _uiState = MutableStateFlow(PlacesUiState())
    val uiState: StateFlow<PlacesUiState> = _uiState.asStateFlow()
    
    init {
        loadHospitals()
    }
    
    fun loadHospitals() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)
            
            placesRepository.getNearbyHospitals()
                .onSuccess { hospitals ->
                    val sortedHospitals = sortHospitals(hospitals)
                    _uiState.value = _uiState.value.copy(
                        hospitals = sortedHospitals,
                        isLoading = false,
                        isEmpty = hospitals.isEmpty()
                    )
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        errorMessage = exception.message ?: "Failed to fetch hospitals."
                    )
                }
        }
    }
    
    fun getCurrentLocation() {
        viewModelScope.launch {
            placesRepository.getCurrentLocationString()
                .onSuccess { locationString ->
                    // Extract coordinates for geocoding
                    val coordinates = locationString.split(",")
                    if (coordinates.size == 2) {
                        val lat = coordinates[0].toDoubleOrNull()
                        val lng = coordinates[1].toDoubleOrNull()
                        
                        if (lat != null && lng != null) {
                            // Get place name from coordinates
                            geocodingService.getPlaceName(lat, lng)
                                .onSuccess { geocodingResult ->
                                    _uiState.value = _uiState.value.copy(
                                        currentLocation = locationString,
                                        currentPlaceName = geocodingResult.placeName,
                                        isLocationPermissionGranted = true
                                    )
                                }
                                .onFailure {
                                    _uiState.value = _uiState.value.copy(
                                        currentLocation = locationString,
                                        currentPlaceName = "Current Location",
                                        isLocationPermissionGranted = true
                                    )
                                }
                        } else {
                            _uiState.value = _uiState.value.copy(
                                currentLocation = locationString,
                                currentPlaceName = "Current Location",
                                isLocationPermissionGranted = true
                            )
                        }
                    } else {
                        _uiState.value = _uiState.value.copy(
                            currentLocation = locationString,
                            currentPlaceName = "Current Location",
                            isLocationPermissionGranted = true
                        )
                    }
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLocationPermissionGranted = false,
                        errorMessage = exception.message ?: "Unable to get current location"
                    )
                }
        }
    }
    
    private fun sortHospitals(hospitals: List<Hospital>): List<Hospital> {
        return hospitals.sortedWith(compareBy<Hospital> { hospital ->
            // First sort by open/closed status (open first)
            when (hospital.openingHours?.openNow) {
                true -> 0
                false -> 1
                null -> 2
            }
        }.thenBy { hospital ->
            // Then sort by distance (nearest first)
            // Extract numeric value from distance string (e.g., "4.2 km" -> 4.2)
            hospital.distance?.let { distanceStr ->
                distanceStr.replace("km", "").replace("m", "").trim().toDoubleOrNull() ?: Double.MAX_VALUE
            } ?: Double.MAX_VALUE
        })
    }
    
    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }
}

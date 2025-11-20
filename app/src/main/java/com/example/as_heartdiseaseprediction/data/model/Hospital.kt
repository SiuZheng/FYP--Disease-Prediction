package com.example.as_heartdiseaseprediction.data.model

import com.google.gson.annotations.SerializedName

data class Hospital(
    @SerializedName("place_id")
    val placeId: String,
    val name: String,
    val vicinity: String?,
    val rating: Double?,
    @SerializedName("user_ratings_total")
    val userRatingsTotal: Int?,
    val geometry: Geometry?,
    val types: List<String>?,
    @SerializedName("opening_hours")
    val openingHours: OpeningHours?,
    @SerializedName("formatted_phone_number")
    val formattedPhoneNumber: String?,
    val website: String?,
    val distance: String? = null,
    val duration: String? = null
)

data class Geometry(
    val location: Location
)

data class Location(
    val lat: Double,
    val lng: Double
)

data class OpeningHours(
    @SerializedName("open_now")
    val openNow: Boolean?,
    val weekdayText: List<String>?
)

data class PlacesResponse(
    val results: List<Hospital>,
    val status: String,
    @SerializedName("error_message")
    val errorMessage: String?
)

data class DistanceMatrixResponse(
    val rows: List<DistanceMatrixRow>,
    val status: String,
    @SerializedName("error_message")
    val errorMessage: String?
)

data class DistanceMatrixRow(
    val elements: List<DistanceMatrixElement>
)

data class DistanceMatrixElement(
    val distance: DistanceValue?,
    val duration: DurationValue?,
    val status: String
)

data class DistanceValue(
    val text: String,
    val value: Int
)

data class DurationValue(
    val text: String,
    val value: Int
)

data class PlaceDetailsResponse(
    val result: PlaceDetailsResult?,
    val status: String,
    @SerializedName("error_message")
    val errorMessage: String?
)

data class PlaceDetailsResult(
    val name: String?,
    @SerializedName("formatted_address")
    val formattedAddress: String?,
    @SerializedName("formatted_phone_number")
    val formattedPhoneNumber: String?,
    @SerializedName("international_phone_number")
    val internationalPhoneNumber: String?,
    val website: String?,
    @SerializedName("opening_hours")
    val openingHours: PlaceDetailsOpeningHours?
)

data class PlaceDetailsOpeningHours(
    @SerializedName("open_now")
    val openNow: Boolean?,
    @SerializedName("weekday_text")
    val weekdayText: List<String>?
)

data class GeocodingResponse(
    val results: List<GeocodingResult>,
    val status: String,
    @SerializedName("error_message")
    val errorMessage: String?
)

data class GeocodingResult(
    @SerializedName("formatted_address")
    val formattedAddress: String?,
    @SerializedName("address_components")
    val addressComponents: List<AddressComponent>?
)

data class AddressComponent(
    @SerializedName("long_name")
    val longName: String,
    val types: List<String>
)

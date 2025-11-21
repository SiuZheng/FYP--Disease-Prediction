package com.example.as_heartdiseaseprediction.data.api

import com.example.as_heartdiseaseprediction.data.model.PlaceDetailsResponse
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

interface PlaceDetailsApiService {
    @GET("place/details/json")
    suspend fun getPlaceDetails(
        @Query("place_id") placeId: String,
        @Query("fields") fields: String,
        @Query("key") key: String
    ): PlaceDetailsResponse
}

object PlaceDetailsApiClient {
    private const val BASE_URL = "https://maps.googleapis.com/maps/api/"
    private const val API_KEY = ""
    
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    
    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
    
    val placeDetailsApiService: PlaceDetailsApiService = retrofit.create(PlaceDetailsApiService::class.java)
}

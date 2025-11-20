package com.example.as_heartdiseaseprediction.data.api

import com.example.as_heartdiseaseprediction.data.model.PlacesResponse
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

interface PlacesApiService {
    @GET("place/nearbysearch/json")
    suspend fun getNearbyHospitals(
        @Query("location") location: String,
        @Query("radius") radius: Int,
        @Query("type") type: String,
        @Query("key") key: String
    ): PlacesResponse
}

object PlacesApiClient {
    private const val BASE_URL = "https://maps.googleapis.com/maps/api/"
    private const val API_KEY = "AIzaSyDHn03_dCpJks2vQ-eRMR_TjQp67bM4UTY"
    
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
    
    val placesApiService: PlacesApiService = retrofit.create(PlacesApiService::class.java)
}

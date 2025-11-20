package com.example.as_heartdiseaseprediction.data.api

import com.example.as_heartdiseaseprediction.data.model.DistanceMatrixResponse
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

interface DistanceMatrixApiService {
    @GET("distancematrix/json")
    suspend fun getDistanceMatrix(
        @Query("origins") origins: String,
        @Query("destinations") destinations: String,
        @Query("key") key: String,
        @Query("units") units: String = "metric"
    ): DistanceMatrixResponse
}

object DistanceMatrixApiClient {
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
    
    val distanceMatrixApiService: DistanceMatrixApiService = retrofit.create(DistanceMatrixApiService::class.java)
}

package com.example.as_heartdiseaseprediction

import android.app.Application
import com.google.firebase.FirebaseApp


class HeartDiseasePredictionApp : Application() {
    override fun onCreate() {
        super.onCreate()
        FirebaseApp.initializeApp(this)
    }
} 
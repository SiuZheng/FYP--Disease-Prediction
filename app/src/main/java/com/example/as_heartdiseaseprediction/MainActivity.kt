package com.example.as_heartdiseaseprediction

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.rememberNavController
import com.example.as_heartdiseaseprediction.data.preferences.OnboardingPreferences
import com.example.as_heartdiseaseprediction.navigation.NavGraph
import com.example.as_heartdiseaseprediction.navigation.Screen
import com.example.as_heartdiseaseprediction.ui.screens.OnboardingScreen
import com.example.as_heartdiseaseprediction.ui.theme.DiseasePredictionTheme
import com.example.as_heartdiseaseprediction.ui.viewmodel.AuthViewModel
import com.example.as_heartdiseaseprediction.ui.viewmodel.PredictionViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DiseasePredictionTheme(dynamicColor = false) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                val context = LocalContext.current
                val onboardingPreferences = remember { OnboardingPreferences(context) }
                var showOnboarding by remember { mutableStateOf(onboardingPreferences.shouldShowOnboarding()) }
                
                val navController = rememberNavController()
                val viewModel: PredictionViewModel = viewModel()
                val authViewModel: AuthViewModel = viewModel()
                
                if (showOnboarding) {
                    OnboardingScreen(
                        onLearnMore = {
                            // This is now handled internally by the onboarding screen
                        },
                        onUnderstand = {
                            showOnboarding = false
                        }
                    )
                } else {
                    NavGraph(
                        navController = navController,
                        viewModel = viewModel
                    )
                }
                }
            }
        }
    }
}
package com.example.as_heartdiseaseprediction.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.as_heartdiseaseprediction.data.model.PredictionRequest
import com.example.as_heartdiseaseprediction.ui.components.BottomNavigationBar
import com.example.as_heartdiseaseprediction.ui.screens.*
import com.example.as_heartdiseaseprediction.ui.viewmodel.AuthViewModel
import com.example.as_heartdiseaseprediction.ui.viewmodel.PredictionViewModel
import com.example.as_heartdiseaseprediction.ui.viewmodel.ChatViewModel
import com.example.as_heartdiseaseprediction.ui.viewmodel.PlacesViewModel

sealed class Screen(val route: String) {
    object Auth : Screen("auth")
    object Home : Screen("home")
    object GeneralUserForm : Screen("general_user_form")
    object HealthcareProfessionalForm : Screen("healthcare_professional_form")
    object Loading : Screen("loading")
    object Results : Screen("results")
    object History : Screen("history")
    object Chatbot : Screen("chatbot")
    object Places : Screen("places")
    object Settings : Screen("settings")
    object About : Screen("about")
}

@Composable
fun NavGraph(
    navController: NavHostController,
    viewModel: PredictionViewModel
) {
    val authViewModel: AuthViewModel = viewModel()
    val isAuthenticated by authViewModel.isAuthenticated.collectAsState()

    // Handle authentication state changes
    LaunchedEffect(isAuthenticated) {
        if (isAuthenticated) {
            // Clear the back stack and navigate to home
            navController.navigate(Screen.Home.route) {
                popUpTo(0) { inclusive = true }
                launchSingleTop = true
            }
        } else {
            // Clear the back stack and navigate to auth
            navController.navigate(Screen.Auth.route) {
                popUpTo(0) { inclusive = true }
                launchSingleTop = true
            }
        }
    }

    NavHost(
        navController = navController,
        startDestination = if (isAuthenticated) Screen.Home.route else Screen.Auth.route
    ) {
        composable(Screen.Auth.route) {
            AuthNavigation(
                onAuthComplete = {
                    // The LaunchedEffect above will handle navigation
                    // when isAuthenticated changes
                }
            )
        }

        composable(Screen.Home.route) {
            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                HomeScreen(
                    onGeneralUserClick = {
                        navController.navigate(Screen.GeneralUserForm.route)
                    },
                    onHealthcareProfessionalClick = {
                        navController.navigate(Screen.HealthcareProfessionalForm.route)
                    },
                    onViewHistoryClick = {
                        navController.navigate(Screen.History.route)
                    },
                    modifier = Modifier.padding(padding)
                )
            }
        }

        composable(Screen.GeneralUserForm.route) {
            GeneralUserFormScreen(
                onBackClick = { navController.navigate(Screen.Home.route) },
                onSubmit = { request ->
                    viewModel.predictHeartDisease(request)
                    navController.navigate(Screen.Results.route)
                },
                viewModel = viewModel
            )
        }

        composable(Screen.HealthcareProfessionalForm.route) {
            HealthcareProfessionalFormScreen(
                onBackClick = { navController.navigate(Screen.Home.route) },
                onSubmit = { request ->
                    viewModel.predictHeartDiseaseDoctor(request)
                    navController.navigate(Screen.Results.route)
                },
                viewModel = viewModel
            )
        }

        composable(Screen.Results.route) {
            val predictionResult by viewModel.predictionResult.collectAsState()
            val isLoading by viewModel.isLoading.collectAsState()
            val username by viewModel.username.collectAsState()
            val shapPlot by viewModel.shapPlot.collectAsState()
            val explanation by viewModel.explanation.collectAsState()
            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                ResultsScreen(
                    predictionResult = predictionResult,
                    isLoading = isLoading,
                    onBackClick = { navController.navigate(Screen.Home.route) },
                    username = username,
                    shapPlot = shapPlot,
                    explanation = explanation,
                    onAboutClick = { 
                        navController.navigate(Screen.About.route) {
                            // Keep the Results screen in the back stack
                            popUpTo(Screen.Results.route) { inclusive = false }
                        }
                    },
                    modifier = Modifier.padding(padding)
                )
            }
        }

        composable(Screen.History.route) {
            val history by viewModel.history.collectAsState()
            val isLoading by viewModel.isLoading.collectAsState()

            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                HistoryScreen(
                    modifier = Modifier.padding(padding),
                    history = history,
                    isLoading = isLoading,
                    onLoadHistory = { viewModel.fetchPredictionHistory() },
                    onBackClick = { navController.popBackStack() }
                )
            }
        }

        composable(Screen.Chatbot.route) {
            val chatViewModel: ChatViewModel = viewModel()
            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                ChatbotScreen(
                    viewModel = chatViewModel,
                    modifier = Modifier.padding(padding)
                )
            }
        }

        composable(Screen.Places.route) {
            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                PlacesScreen(
                    modifier = Modifier.padding(padding)
                )
            }
        }

        composable(Screen.Settings.route) {
            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                SettingsScreen(
                    onBackClick = {
                        navController.navigate(Screen.Home.route)
                    },
                    modifier = Modifier.padding(padding)
                )
            }
        }

        composable(Screen.About.route) {
            Scaffold(
                bottomBar = {
                    BottomNavigationBar(navController = navController)
                }
            ) { padding ->
                AboutScreen(
                    onBackClick = { navController.popBackStack() },
                    modifier = Modifier.padding(padding)
                )
            }
        }
    }
} 
package com.example.as_heartdiseaseprediction.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.as_heartdiseaseprediction.ui.screens.SignInScreen
import com.example.as_heartdiseaseprediction.ui.screens.SignUpScreen

sealed class AuthScreen(val route: String) {
    object SignIn : AuthScreen("sign_in")
    object SignUp : AuthScreen("sign_up")
}

@Composable
fun AuthNavigation(
    navController: NavHostController = rememberNavController(),
    onAuthComplete: () -> Unit
) {
    NavHost(
        navController = navController,
        startDestination = AuthScreen.SignIn.route
    ) {
        composable(AuthScreen.SignIn.route) {
            SignInScreen(
                onAuthComplete = onAuthComplete,
                onSignUpClick = { navController.navigate(AuthScreen.SignUp.route) }
            )
        }
        
        composable(AuthScreen.SignUp.route) {
            SignUpScreen(
                onAuthComplete = onAuthComplete,
                onSignInClick = { navController.navigate(AuthScreen.SignIn.route) }
            )
        }
    }
}
package com.example.as_heartdiseaseprediction.ui.components

import androidx.compose.foundation.Image
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import com.example.as_heartdiseaseprediction.R
import com.example.as_heartdiseaseprediction.navigation.Screen
import androidx.compose.foundation.layout.size
@Composable
fun BottomNavigationBar(navController: NavController) {
    val items = listOf(
        Screen.Home,
        Screen.Chatbot,
        Screen.Places,
        Screen.Settings,
        Screen.About
    )

    NavigationBar {
        val navBackStackEntry = navController.currentBackStackEntryAsState()
        val currentRoute = navBackStackEntry.value?.destination?.route

        items.forEach { screen ->
            NavigationBarItem(
                icon = {
                    when (screen) {
                        Screen.Home -> Icon(Icons.Default.Home, contentDescription = screen.route)
                        Screen.Chatbot -> Image(
                            painter = painterResource(id = R.drawable.robot_icon),
                            contentDescription = screen.route,
                            modifier = Modifier.size(24.dp)
                        )
                        Screen.Places -> Icon(
                            Icons.Default.LocationOn,
                            contentDescription = screen.route
                        )
                        Screen.Settings -> Icon(
                            Icons.Default.Settings,
                            contentDescription = screen.route
                        )
                        Screen.About -> Icon(Icons.Default.Info, contentDescription = screen.route)
                        else -> Icon(Icons.Default.Home, contentDescription = screen.route)
                    }
                },
                label = { Text(screen.route) },
                selected = currentRoute == screen.route,
                onClick = {
                    navController.navigate(screen.route) {
                        popUpTo(navController.graph.startDestinationId)
                        launchSingleTop = true
                    }
                }
            )
        }
    }
} 